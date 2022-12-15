#!/bin/sh
#BSUB -J VAE_beta_4
#BSUB -o VAE%J.out
#BSUB -e VAE%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -W 24:00
#BSUB -N
# end of BSUB options

module load python3/3.9.6

# load CUDA (for GPU support)
module load cuda/11.7

# activate the virtual environment
source DGM/bin/activate

python main.py --seed 3407 --device cuda --write --mode "traintest" --experiment "bank" --dataset "bank" --scale "normalize" --max_epochs 500 --max_patience 100 --prior "vampPrior" --beta 0.01
