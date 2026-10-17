#!/bin/bash
# usage (from this directory): bash run_all.sh ada|pro <minutes_per_model> [repeats]
# The "pro" grid (bigger batch/context for a 48 GB GPU) has not been run yet.
PRESET=${1:-ada}; MIN=${2:-20}; REP=${3:-3}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_OFFLINE=1

# Override these if your model paths differ.
GQA_MODEL=${GQA_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
MHA_MODEL=${MHA_MODEL:-$HOME/models/open_llama_3b_v2}
MODELS=("gqa_3b:$GQA_MODEL" "mha_3b:$MHA_MODEL")

python calibrate.py || exit 1
for spec in "${MODELS[@]}"; do
  tag=${spec%%:*}; mid=${spec#*:}
  echo "=== $tag ($mid) ==="
  python bench_decode.py --model "$mid" --model_tag "$tag" --preset "$PRESET" \
         --max_minutes "$MIN" --repeats "$REP"
done
python summarize.py