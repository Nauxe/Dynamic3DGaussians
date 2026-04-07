#!/bin/bash
#SBATCH -J d3dg 
#SBATCH --gres=gpu:a100-40:1

#SBATCH --time=05:00:00 # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --output="slurm-out/train-render-%j.out"

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=

source ~/.bashrc 

conda activate 3dg

# python train.py
python train.py 

python render.py --exp-name sphere-bounce-5

python plot_losses.py

cd output/sphere-bounce-5/sphere-bounce-5/test/ours-625x625/renders
# convert -delay 20 -loop 0 *0012.png myimage.gif
ffmpeg -f image2 -r 5 -pattern_type glob -i '*0012.png' -vcodec libx264 -crf 22 video.mp4 
mv video.mp4 ../../../renders-12.mp4
