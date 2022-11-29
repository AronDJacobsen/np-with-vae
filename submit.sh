#!/bin/sh
#BSUB -J VAE_avocado
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
module load cuda/11.3

# activate the virtual environment
source DGM/bin/activate

python main.py --seed 42 --device cuda --batch 16 --write