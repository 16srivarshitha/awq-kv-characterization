import argparse, json, math, os, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--ctx", type=int, nargs="+", default=[2048, 4096])
p.add_argument("--bits", type=int, nargs="+", default=[16, 8])
p.add_argument("--group", type=int, default=128, help="quantization group along head_dim (128 = per token per head)")
p.add_argument("--max_windows", type=int, default=64)
p.add_argument("--out", default="results/ppl_int8_kv.json")
a = p.parse_args()

CUR, CALLS = {"bits": 16}, [0]

def fq(x, bits, group):
    if bits >= 16:
        return x
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group).float()
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((g / scale).round().clamp(-qmax, qmax) * scale).reshape(s).to(x.dtype)

def fakeq_attn(module, query, key, value, attention_mask, **kw):
    CALLS[0] += 1       # key/value here are post-RoPE and un-expanded, i.e. what the cache stores
    return sdpa_attention_forward(module, query, fq(key, CUR["bits"], a.group),
                                  fq(value, CUR["bits"], a.group), attention_mask, **kw)

AttentionInterface.register("fakeq", fakeq_attn)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="fakeq",
                                             device_map="cuda").eval()
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[0]
print(f"{len(ids)} tokens; head_dim {model.config.hidden_size // model.config.num_attention_heads}; "
      f"group {a.group}", flush=True)

def ppl(L, bits):
    CUR["bits"] = bits
    CALLS[0] = 0
    nll, n = 0.0, 0
    nwin = min(len(ids) // L, a.max_windows)
    with torch.no_grad():
        for w in range(nwin):
            x = ids[w * L:(w + 1) * L].unsqueeze(0).cuda()
            h = model.model(input_ids=x, use_cache=False).last_hidden_state
            for s in range(0, L - 1, 512):         # chunked lm_head keeps logits memory small
                e = min(s + 512, L - 1)
                lg = model.lm_head(h[:, s:e]).float()
                nll += torch.nn.functional.cross_entropy(lg[0], x[0, s + 1:e + 1], reduction="sum").item()
            n += L - 1
    return math.exp(nll / n), nwin, CALLS[0]

res = []
for L in a.ctx:
    base = None
    for bits in a.bits:
        t = time.time()
        v, nwin, calls = ppl(L, bits)
        base = v if bits == 16 else base
        print(f"ctx {L:5d} KV bits {bits:2d} | ppl {v:8.4f} | delta vs fp16 {v - base:+.4f} | "
              f"{nwin} windows | fakeq calls {calls} | {time.time() - t:.0f}s", flush=True)
        res.append(dict(ctx=L, bits=bits, ppl=v, windows=nwin, group=a.group, fakeq_calls=calls))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
