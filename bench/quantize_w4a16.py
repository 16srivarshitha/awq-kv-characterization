import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUT = os.path.expanduser("~/models/llama32-3b-w4a16-g128")
N, L = 128, 512

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")

ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
ds = ds.shuffle(seed=0).select(range(N))
ds = ds.map(lambda x: tok(x["text"], padding=False, max_length=L, truncation=True,
                          add_special_tokens=False), remove_columns=ds.column_names)

recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=L, num_calibration_samples=N)

model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("QUANT_OK", OUT)
