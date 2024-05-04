#!/bin/bash -l
#SBATCH --gres=gpu:rtx3080:1 --partition=rtx3080 --time=24:00:00
#SBATCH --job-name=CONV3_TRAINING
#SBATCH --export=NONE

source ~/.bashrc
module load cuda
conda activate hdc
cd /home/hpc/iwi3/iwi3083h/network-ecg/hdc_train

python train_conv.py