#!/bin/bash -l
#SBATCH --gres=gpu:rtx3080:1 --partition=rtx3080 --time=24:00:00
#SBATCH --job-name=MNIST_HDC
#SBATCH --export=NONE

source ~/.bashrc
conda activate hdc
module load cuda

rm -rf ~/.openhd
cd /home/hpc/iwi3/iwi3083h/hdc_dnn



python MNIST-HDC.py
