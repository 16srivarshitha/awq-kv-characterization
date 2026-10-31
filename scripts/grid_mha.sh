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
python bench_vllm.py --model ~/models/open_llama_3b_v2 --model_tag mha3b_fp16 \
       --ctx 128 512 1024 1792 --outdir results_vllm_mha
echo GRID_OK
