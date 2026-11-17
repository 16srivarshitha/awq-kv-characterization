import os
import sys
import csv
import math
import time
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def make_randomized_hadamard_128(device, dtype=torch.float32, seed=42):
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    for _ in range(6):
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    h = h / math.sqrt(128.0)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (128,), generator=gen, dtype=torch.int32) * 2 - 1).to(device=device, dtype=torch.float32)
    return (signs[:, None] * h).to(dtype)


def quant_sym_int8(x, scale):
    return torch.round(x / scale).clamp(-128, 127) * scale


def quant_per_token_int8(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return quant_sym_int8(x, s)


STATE = {
    "mode": "fp16",
    "h_mat": None,
    "calib_k_raw": {},
    "calib_k_rot": {},
    "calib_v_chan": {},
    "outlier_stats": [],
}


def hadamard_attn_interface(module, query, key, value, attention_mask, **kw):
    mode = STATE["mode"]
    if mode == "fp16":
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    lidx = module.layer_idx
    h_mat = STATE["h_mat"]
    g = query.shape[1] // key.shape[1]

    if mode == "calibrate":
        kf = key.float()
        vf = value.float()
        k_rot = torch.matmul(kf, h_mat)

        # update running abs-max across calibration windows
        cur_raw = kf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
        cur_rot = k_rot.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
        cur_v = vf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0

        if lidx not in STATE["calib_k_raw"]:
            STATE["calib_k_raw"][lidx] = cur_raw
            STATE["calib_k_rot"][lidx] = cur_rot
            STATE["calib_v_chan"][lidx] = cur_v
        else:
            STATE["calib_k_raw"][lidx] = torch.maximum(STATE["calib_k_raw"][lidx], cur_raw)
            STATE["calib_k_rot"][lidx] = torch.maximum(STATE["calib_k_rot"][lidx], cur_rot)
            STATE["calib_v_chan"][lidx] = torch.maximum(STATE["calib_v_chan"][lidx], cur_v)

        raw_ratio = (kf.abs().amax(2).max(-1).values / kf.abs().amax(2).median(-1).values).max().item()
        rot_ratio = (k_rot.abs().amax(2).max(-1).values / k_rot.abs().amax(2).median(-1).values).max().item()
        STATE["outlier_stats"].append((lidx, raw_ratio, rot_ratio))
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    if mode == "no_had_static":
        k_q = quant_sym_int8(key.float(), STATE["calib_k_raw"][lidx]).to(key.dtype)
        v_q = quant_sym_int8(value.float(), STATE["calib_v_chan"][lidx]).to(value.dtype)
        return sdpa_attention_forward(module, query, k_q, v_q, attention_mask, **kw)

    if mode == "no_had_kchan_vtok":
        k_q = quant_sym_int8(key.float(), STATE["calib_k_raw"][lidx]).to(key.dtype)
        v_q = quant_per_token_int8(value.float()).to(value.dtype)
        return sdpa_attention_forward(module, query, k_q, v_q, attention_mask, **kw)

    if mode == "had_kv8_a16_static":
        q_rot = torch.matmul(query.float(), h_mat).to(query.dtype)
        k_rot = torch.matmul(key.float(), h_mat)
        k_q = quant_sym_int8(k_rot, STATE["calib_k_rot"][lidx]).to(key.dtype)
        v_q = quant_sym_int8(value.float(), STATE["calib_v_chan"][lidx]).to(value.dtype)
        return sdpa_attention_forward(module, q_rot, k_q, v_q, attention_mask, **kw)

    if mode == "had_kv8_a16_vtok":
        q_rot = torch.matmul(query.float(), h_mat).to(query.dtype)
        k_rot = torch.matmul(key.float(), h_mat)
        k_q = quant_sym_int8(k_rot, STATE["calib_k_rot"][lidx]).to(key.dtype)
        v_q = quant_per_token_int8(value.float()).to(value.dtype)
        return sdpa_attention_forward(module, q_rot, k_q, v_q, attention_mask, **kw)

    if mode == "had_kv8_q8_vtok":
        # exact dp4a math: quantize q_tilde per token and k_rot per channel
        s_k = STATE["calib_k_rot"][lidx]
        q_rot = torch.matmul(query.float(), h_mat)
        k_rot = torch.matmul(key.float(), h_mat)

        q_tilde = q_rot * s_k.repeat_interleave(g, dim=1)
        s_q = q_tilde.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q_int8 = torch.round(q_tilde / s_q).clamp(-128, 127)
        k_int8 = torch.round(k_rot / s_k).clamp(-128, 127)

        # rescale symmetrically by 127.0 so both inputs stay in normal fp16 range
        q_eff = (q_int8 * (s_q * 127.0)).to(query.dtype)
        k_eff = (k_int8 / 127.0).to(key.dtype)
        v_q = quant_per_token_int8(value.float()).to(value.dtype)
        return sdpa_attention_forward(module, q_eff, k_eff, v_q, attention_mask, **kw)

    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def eval_ppl(model, ids, ctx_len, max_windows):
    nll, n = 0.0, 0
    nwin = min(len(ids) // ctx_len, max_windows)
    for w in range(nwin):
        x = ids[w * ctx_len : (w + 1) * ctx_len].unsqueeze(0).cuda()
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        for s in range(0, ctx_len - 1, 512):
            e = min(s + 512, ctx_len - 1)
            lg = model.lm_head(h[:, s:e]).float()
            nll += torch.nn.functional.cross_entropy(lg[0], x[0, s + 1 : e + 1], reduction="sum").item()
        n += ctx_len - 1
    return math.exp(nll / n), nwin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--calib_windows", type=int, default=4)
    args = ap.parse_args()

    AttentionInterface.register("hadamard_eval", hadamard_attn_interface)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="hadamard_eval", device_map="cuda"
    ).eval()

    val_text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"])
    test_text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])

    val_ids = tok(val_text, return_tensors="pt").input_ids[0]
    test_ids = tok(test_text, return_tensors="pt").input_ids[0]
    print(f"\nwikitext-2 full evaluation: test_tokens={len(test_ids)}, calib_tokens={len(val_ids)}, ctx={args.ctx}, max_windows={args.max_windows}", flush=True)

    STATE["h_mat"] = make_randomized_hadamard_128("cuda", dtype=torch.float32, seed=42)

    # calibrate static channel scales strictly on validation split
    STATE["mode"] = "calibrate"
    _, cwin = eval_ppl(model, val_ids, args.ctx, args.calib_windows)
    raw_worst = max(x[1] for x in STATE["outlier_stats"])
    rot_worst = max(x[2] for x in STATE["outlier_stats"])
    raw_avg = sum(x[1] for x in STATE["outlier_stats"]) / len(STATE["outlier_stats"])
    rot_avg = sum(x[2] for x in STATE["outlier_stats"]) / len(STATE["outlier_stats"])
    print(f"calibrated static scales on {cwin} validation windows ({cwin * args.ctx} tokens):", flush=True)
    print(f"  key outlier max/median ratio: pre-hadamard avg={raw_avg:.2f}x worst={raw_worst:.2f}x | post-hadamard avg={rot_avg:.2f}x worst={rot_worst:.2f}x\n", flush=True)

    modes = [
        ("fp16",               "fp16 baseline (eval 1 anchor)"),
        ("no_had_static",      "no-hadamard static (k,v)"),
        ("no_had_kchan_vtok",  "no-hadamard static k + tok v"),
        ("had_kv8_a16_static", "rand-hadamard static (k,v)"),
        ("had_kv8_a16_vtok",   "rand-hadamard stat k + tok v (triton)"),
        ("had_kv8_q8_vtok",    "rand-hadamard q8+k_stat+v_tok (dp4a)"),
    ]

    csv_rows = []
    fp16_ppl = None

    for m, desc in modes:
        STATE["mode"] = m
        t0 = time.time()
        val, nwin = eval_ppl(model, test_ids, args.ctx, args.max_windows)
        if fp16_ppl is None:
            fp16_ppl = val
        delta = val - fp16_ppl
        dt = time.time() - t0
        print(f"  {desc:<40} | ppl: {val:8.4f} | delta: {delta:+.4f} | {nwin} wins | {dt:.1f}s", flush=True)
        csv_rows.append([m, desc, args.ctx, nwin, f"{val:.4f}", f"{delta:+.4f}"])

    csv_path = os.path.join(repo_root, "results", "csv", "wikitext2_full_ppl.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "description", "ctx", "windows", "ppl", "delta_vs_fp16"])
        w.writerows(csv_rows)
    print(f"\nsaved full wikitext-2 ppl results to {csv_path}\n", flush=True)


if __name__ == "__main__":
    main()
