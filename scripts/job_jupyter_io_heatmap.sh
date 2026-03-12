#!/bin/bash
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -p gpu_requeue
#SBATCH --mem=128000
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/plot-%j.out

source ~/.bashrc
conda activate dynamic
python heatmap_incl_a.py \
    --orbits 1000 \
    --name jupyter_io_heatmap_more \
    --divisor 2 \
    --a-p 7.78e11 \
    --M-p 1.989e30 \
    --r-p 7.15e7 \
    --M-m 1.898e27 \
    --a-m-steps 100 \
    --a-m-min 2e8 \
    --a-m-max 1e10
