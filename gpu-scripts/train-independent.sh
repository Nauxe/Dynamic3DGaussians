#!/bin/bash
#SBATCH -J d3dg-independent 
#SBATCH --gres=gpu:a100-40:1

#SBATCH --time=24:00:00 # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --output="slurm-out/train-independent-%j.out"

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=$SLURM_EMAIL

source ~/.bashrc 

conda activate 3dg

python train_independent.py
python render_independent.py
