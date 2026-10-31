#!/bin/bash
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH -c 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH -o %x_%j.log
set -euo pipefail
source ~/vllm312/bin/activate
cd ~/awq-kv-characterization/bench
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python bench_vllm.py --model meta-llama/Llama-3.2-3B-Instruct --model_tag gqa3b_fp16
python bench_vllm.py --model ~/models/llama32-3b-w4a16-g128 \
       --plan_model meta-llama/Llama-3.2-3B-Instruct --model_tag gqa3b_w4a16
echo GRID_OK
