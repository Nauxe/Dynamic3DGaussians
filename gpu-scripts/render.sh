#!/bin/bash
#SBATCH -J d3dg 
#SBATCH --gres=gpu:a100-40:1

#SBATCH --time=05:00:00 # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --output="slurm-out/render-%j.out"

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=$SLURM_EMAIL

source ~/.bashrc 

conda activate 3dg

python render.py
