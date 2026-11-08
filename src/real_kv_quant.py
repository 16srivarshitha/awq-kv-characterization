
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = []


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def q_tok(x):
    """INT8 symmetric, scale per (head, token) over channels. x [H,S,D]."""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def q_chan(x, calib):
    """INT8 symmetric, static scale per (head, channel) from calibration tokens."""
    s = x[:, :calib].abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def attn_out(qs, k, v, S, nq):
    HQ, _, D = qs.shape
    G = HQ // k.shape[0]
    kk = k.repeat_interleave(G, 0)
    vv = v.repeat_interleave(G, 0)
    s = torch.einsum("hqd,hsd->hqs", qs, kk) / D ** 0.5
    pos = torch.arange(S - nq, S, device=qs.device)
    mask = torch.arange(S, device=qs.device)[None, :] > pos[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqs,hsd->hqd", s.softmax(-1), vv)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@torch.no_grad()
def analyze(layer, q, k, v, nq):
    HQ, S, D = q.shape
    qs = q[:, S - nq:, :]
    ref = attn_out(qs, k, v, S, nq)
    vq = q_tok(v)
    o_tok = attn_out(qs, q_tok(k), vq, S, nq)
    o_chan = attn_out(qs, q_chan(k, S // 2), vq, S, nq)
    a = k.abs().amax(1)                                  # [HKV, D] per-channel abs-max
    ratio = (a.max(-1).values / a.median(-1).values)
    RESULTS.append(dict(
        layer=layer,
        outlier_mean=ratio.mean().item(), outlier_max=ratio.max().item(),
        k_err_tok=rel(q_tok(k), k), k_err_chan=rel(q_chan(k, S // 2), k),
        out_err_tok=rel(o_tok, ref), out_err_chan=rel(o_chan, ref),
        v_err=rel(vq, v)))


def make_hook(nq):
    def hook(module, args, kwargs, output):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings")
        if hs is None or pe is None:
            raise RuntimeError("hook could not find hidden_states/position_embeddings in the "
                               "attention call; this transformers version needs a small tweak")
        cfg = module.config
        HQ = cfg.num_attention_heads
        HKV = cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // HQ
        S = hs.shape[1]
        cos, sin = pe
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)    # [1,1,S,D]
        q = module.q_proj(hs).view(1, S, HQ, D).transpose(1, 2)
        k = module.k_proj(hs).view(1, S, HKV, D).transpose(1, 2)
        v = module.v_proj(hs).view(1, S, HKV, D).transpose(1, 2)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        analyze(module.layer_idx, q[0].float(), k[0].float(), v[0].float(), nq)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--nq", type=int, default=64, help="number of final query positions to evaluate")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    ids = tok(open(a.text_file).read(), return_tensors="pt").input_ids
    assert ids.shape[1] >= a.seq, f"text has only {ids.shape[1]} tokens, need {a.seq}; use a longer file"
    ids = ids[:, :a.seq].cuda()

    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(make_hook(a.nq), with_kwargs=True)
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)       # base model: no lm_head, no logits

    print(f"{'L':>2} | {'K outlier mean/max':>18} | {'K err tok':>9} {'K err chan':>10} | "
          f"{'V err':>6} | {'attn-out err tok':>16} {'chan':>7}")
    for r in sorted(RESULTS, key=lambda r: r["layer"]):
        print(f"{r['layer']:>2} | {r['outlier_mean']:>8.1f} /{r['outlier_max']:>8.1f} | "
              f"{r['k_err_tok']:>9.4f} {r['k_err_chan']:>10.4f} | {r['v_err']:>6.4f} | "
              f"{r['out_err_tok']:>16.5f} {r['out_err_chan']:>7.5f}")
    n = len(RESULTS)
    avg = lambda key: sum(r[key] for r in RESULTS) / n
    print(f"\nmean over {n} layers: attn-out err  tok={avg('out_err_tok'):.5f}  chan={avg('out_err_chan'):.5f}"
          f"   worst layer tok={max(r['out_err_tok'] for r in RESULTS):.5f}"
          f"  chan={max(r['out_err_chan'] for r in RESULTS):.5f}")


if __name__ == "__main__":
    main()
