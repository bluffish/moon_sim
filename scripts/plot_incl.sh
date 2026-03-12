#!/bin/bash
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p seas_gpu
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --output=./outputs/plot-%j.out

source ~/.bashrc

conda activate dynamic 
python plot_eclipse_vs_incl.py
