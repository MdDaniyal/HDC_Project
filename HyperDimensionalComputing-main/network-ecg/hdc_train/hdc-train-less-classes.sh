#!/bin/bash -l
#SBATCH --gres=gpu:rtx3080:1 --partition=rtx3080 --time=24:00:00
#SBATCH --job-name=HDC_TRAINING_LESS_CLASSES
#SBATCH --export=NONE

source ~/.bashrc
module load cuda
conda activate hdc
cd /home/hpc/iwi3/iwi3083h/network-ecg/hdc_train

# # SNR - AF
# python hdc_train.py --classes SNR,AF --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# # SNR - AF - RBBB
# python hdc_train.py --classes SNR,AF,RBBB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes

# # SNR - AF - RBBB - LBBB
# python hdc_train.py --classes SNR,AF,RBBB,LBBB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB,LBBB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# # SNR - AF - RBBB - IAVB
# python hdc_train.py --classes SNR,AF,RBBB,IAVB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB,IAVB --use-ngram --enls
# coding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# # SNR - AF - RBBB - STD
# python hdc_train.py --classes SNR,AF,RBBB,STD --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB,STD --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# # SNR - AF - RBBB - PVC
# python hdc_train.py --classes SNR,AF,RBBB,PVC --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB,PVC --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# # SNR - AF - RBBB - LBBB - IAVB
# python hdc_train.py --classes SNR,AF,RBBB,IAVB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes SNR,AF,RBBB,IAVB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes

# # all
# python hdc_train.py --classes all --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes
# python hdc_train.py --classes all --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes


# SNR - AF
python hdc_train.py --epoch 10 --classes SNR,AF --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --epoch 10 --classes SNR,AF --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


# SNR - AF - RBBB
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10

# SNR - AF - RBBB - LBBB
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,LBBB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,LBBB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


# SNR - AF - RBBB - IAVB
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,IAVB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,IAVB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


# SNR - AF - RBBB - STD
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,STD --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,STD --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


# SNR - AF - RBBB - PVC
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,PVC --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,PVC --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


# SNR - AF - RBBB - LBBB - IAVB
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,IAVB --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes SNR,AF,RBBB,IAVB --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10

# all
python hdc_train.py --epoch 10 --classes all --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10
python hdc_train.py --epoch 10 --classes all --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/less-classes-epochs10


