import os
import sys
import csv
import math
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_rand_hadamard(device, seed=42):
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    for _ in range(6):
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    h = h / math.sqrt(128.0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (128,), generator=gen, dtype=torch.int32) * 2 - 1).to(device=device, dtype=torch.float32)
    return signs[:, None] * h


STATE = {"mode": "fp16", "calib_s_k": {}, "h_mat": None}


def quant_sym(x, scale):
    return torch.round(x / scale).clamp(-128, 127) * scale


def quant_tok(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return quant_sym(x, s)


def custom_attn(module, query, key, value, attention_mask, **kw):
    mode = STATE["mode"]
    lidx = module.layer_idx
    h_mat = STATE["h_mat"]

    if mode == "fp16":
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    if mode == "calibrate":
        k_f = key.float()
        k_rot = torch.matmul(k_f, h_mat)
        s_raw = k_f.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
        s_rot = k_rot.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
        if lidx not in STATE["calib_s_k"]:
            STATE["calib_s_k"][lidx] = {"raw": s_raw, "rot": s_rot}
        else:
            STATE["calib_s_k"][lidx]["raw"] = torch.maximum(STATE["calib_s_k"][lidx]["raw"], s_raw)
            STATE["calib_s_k"][lidx]["rot"] = torch.maximum(STATE["calib_s_k"][lidx]["rot"], s_rot)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    g = query.shape[1] // key.shape[1]
    v_q = quant_tok(value.float()).to(value.dtype)

    if mode == "no_had_kstat_vtok":
        s_k = STATE["calib_s_k"][lidx]["raw"]
        q_eff = query
        k_eff = quant_sym(key.float(), s_k).to(key.dtype)
    elif mode == "had_kstat_vtok":
        s_k = STATE["calib_s_k"][lidx]["rot"]
        q_eff = torch.matmul(query.float(), h_mat).to(query.dtype)
        k_eff = quant_sym(torch.matmul(key.float(), h_mat), s_k).to(key.dtype)
    elif mode == "no_had_q8_vtok":
        s_k = STATE["calib_s_k"][lidx]["raw"]
        q_tilde = query.float() * s_k.repeat_interleave(g, dim=1)
        s_q = q_tilde.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q_eff = (torch.round(q_tilde / s_q).clamp(-128, 127) * s_q).to(query.dtype)
        k_eff = torch.round(key.float() / s_k).clamp(-128, 127).to(key.dtype)
    elif mode == "had_q8_vtok":
        s_k = STATE["calib_s_k"][lidx]["rot"]
        q_rot = torch.matmul(query.float(), h_mat)
        k_rot = torch.matmul(key.float(), h_mat)
        q_tilde = q_rot * s_k.repeat_interleave(g, dim=1)
        s_q = q_tilde.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q_eff = (torch.round(q_tilde / s_q).clamp(-128, 127) * s_q).to(query.dtype)
        k_eff = torch.round(k_rot / s_k).clamp(-128, 127).to(key.dtype)
    else:
        raise ValueError(mode)

    return sdpa_attention_forward(module, q_eff, k_eff, v_q, attention_mask, **kw)


@torch.no_grad()
def eval_window_nlls(model, ids, ctx=2048, max_wins=64):
    nwin = min(len(ids) // ctx, max_wins)
    win_nlls = []
    for w in range(nwin):
        x = ids[w * ctx : (w + 1) * ctx].unsqueeze(0).cuda()
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        nll = 0.0
        for s in range(0, ctx - 1, 512):
            e = min(s + 512, ctx - 1)
            lg = model.lm_head(h[:, s:e]).float()
            nll += torch.nn.functional.cross_entropy(lg[0], x[0, s + 1 : e + 1], reduction="sum").item()
        win_nlls.append(nll / (ctx - 1))
    return torch.tensor(win_nlls, dtype=torch.float64)


def main():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    AttentionInterface.register("se_check", custom_attn)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                 attn_implementation="se_check", device_map="cuda").eval()
    STATE["h_mat"] = make_rand_hadamard("cuda", seed=42)

    val_txt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"])
    test_txt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    val_ids = tok(val_txt, return_tensors="pt").input_ids[0]
    test_ids = tok(test_txt, return_tensors="pt").input_ids[0]

    STATE["mode"] = "calibrate"
    _ = eval_window_nlls(model, val_ids, ctx=2048, max_wins=4)

    modes = ["fp16", "no_had_kstat_vtok", "had_kstat_vtok", "no_had_q8_vtok", "had_q8_vtok"]
    nll_map = {}
    for m in modes:
        STATE["mode"] = m
        t0 = time.time()
        nll_map[m] = eval_window_nlls(model, test_ids, ctx=2048, max_wins=64)
        ppl = math.exp(nll_map[m].mean().item())
        print(f"  {m:<20} | ppl: {ppl:.4f} | time: {time.time()-t0:.1f}s")

    print("\npaired delta vs fp16 across 64 windows:")
    base_nll = nll_map["fp16"]
    base_ppl = math.exp(base_nll.mean().item())
    for m in modes[1:]:
        dnll = nll_map[m] - base_nll
        mean_dnll = dnll.mean().item()
        se_dnll = dnll.std(unbiased=True).item() / math.sqrt(len(dnll))
        dppl = math.exp(nll_map[m].mean().item()) - base_ppl
        se_ppl = base_ppl * se_dnll
        print(f"  {m:<20} | delta_ppl: {dppl:+.5f} +/- {se_ppl:.5f} (se) | delta_nll: {mean_dnll:+.5f} +/- {se_dnll:.5f}")


if __name__ == "__main__":
    main()
