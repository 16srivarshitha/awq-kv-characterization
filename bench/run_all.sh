#!/bin/bash
# usage: bash run_all.sh ada|pro <minutes_per_model> [repeats]
PRESET=${1:-ada}; MIN=${2:-20}; REP=${3:-3}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_OFFLINE=1

if [ "$PRESET" = "pro" ]; then
  MODELS=("gqa_3b:meta-llama/Llama-3.2-3B-Instruct" "mha_7b:$HOME/models/llama2_7b")
else
  MODELS=("gqa_3b:meta-llama/Llama-3.2-3B-Instruct" "mha_3b:$HOME/models/open_llama_3b_v2")
fi

python calibrate.py || exit 1
for spec in "${MODELS[@]}"; do
  tag=${spec%%:*}; mid=${spec#*:}
  echo "=== $tag ($mid) ==="
  python bench_decode.py --model "$mid" --model_tag "$tag" --preset "$PRESET" \
         --max_minutes "$MIN" --repeats "$REP"
done
python summarize.py
