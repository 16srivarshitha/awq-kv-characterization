#!/bin/bash
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=low
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
source ~/llm_env/bin/activate
cd ~/bench
bash run_all.sh ada 25 3
