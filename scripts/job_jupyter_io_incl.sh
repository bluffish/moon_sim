#!/bin/bash
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p seas_gpu
#SBATCH --mem=128000
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/plot-%j.out

source ~/.bashrc
conda activate dynamic
python plot_incl.py \
    --orbits 1000 \
    --divisor 2 \
    --window 48 \
    --a-p 7.78e11 \
    --M-p 1.989e30 \
    --r-p 7.15e7 \
    --a-m 4.22e8 \
    --M-m 1.898e27 \
    --name jupyter_io_incl
