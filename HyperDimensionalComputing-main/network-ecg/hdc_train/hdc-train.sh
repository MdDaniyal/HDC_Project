#!/bin/bash -l
#SBATCH --gres=gpu:rtx3080:1 --partition=rtx3080 --time=24:00:00
#SBATCH --job-name=HDC_TRAINING_ngrams
#SBATCH --export=NONE

source ~/.bashrc
module load cuda
conda activate hdc
cd /home/hpc/iwi3/iwi3083h/network-ecg/hdc_train

# python hdc_train.py --num-levels 1000 --sampling 900   >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 1800 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 2700 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 3600 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 4500 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 5400 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 6300 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 7200 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 8100 >> num-levels-1000
# python hdc_train.py --num-levels 1000 --sampling 9000 >> num-levels-1000


# python hdc_train.py --num-levels 1000 --sampling 900 --add-online  >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 1800 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 2700 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 3600 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 4500 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 5400 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 6300 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 7200 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 8100 --add-online >> num-levels-1000-online
# python hdc_train.py --num-levels 1000 --sampling 9000 --add-online >> num-levels-1000-online


# python hdc_train.py --num-levels 16 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 32 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 64 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 128 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 256 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 512 --sampling 9000 >> sampling-9000
# python hdc_train.py --num-levels 1024 --sampling 9000 >> sampling-9000


# python hdc_train.py --num-levels 16 --sampling 9000 --add-online  >> sampling-9000-online
# python hdc_train.py --num-levels 32 --sampling 9000 --add-online >> sampling-9000-online
# python hdc_train.py --num-levels 64 --sampling 9000 --add-online >> sampling-9000-online
# python hdc_train.py --num-levels 128 --sampling 9000 --add-online >> sampling-9000-online
# python hdc_train.py --num-levels 256 --sampling 9000 --add-online >> sampling-9000-online
# python hdc_train.py --num-levels 512 --sampling 9000 --add-online >> sampling-9000-online
# python hdc_train.py --num-levels 1024 --sampling 9000 --add-online >> sampling-9000-online

# python hdc_train.py --epoch 10 --num-levels 32 --sampling 9000 --add-online >> sampling-9000-epochs-10-online
# python hdc_train.py --epoch 1 --num-levels 32 --sampling 900 --add-online --max-val 1000 --min-val -1000 >> ./logs/sampling-90-epochs-max-min-1000



# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 1 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 2 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 4 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 8 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 16 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 32 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 64 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams
# python hdc_train.py --num-levels 32 --sampling 900 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/varying-ngrams


# python hdc_train.py --encoding fv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000
# python hdc_train.py --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000
# python hdc_train.py --encoding fChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000
# python hdc_train.py --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000
# python hdc_train.py --encoding chFeatVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000

# python hdc_train.py --use-ngram --encoding fv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000-ngram
# python hdc_train.py --use-ngram --encoding cv --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000-ngram
# python hdc_train.py --use-ngram --encoding fChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000-ngram
# python hdc_train.py --use-ngram --encoding featChVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000-ngram
# python hdc_train.py --use-ngram --encoding chFeatVal --num-levels 32 --sampling 9000 --add-online --n-gram 128 --max-val 1000 --min-val -1000 >> ./logs/encodings-9000-ngram



