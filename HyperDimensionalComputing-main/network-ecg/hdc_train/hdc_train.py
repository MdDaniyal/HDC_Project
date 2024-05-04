from utils import parse_args
import os
from pathlib import Path as pl

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wfdb
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.datasets import EMGHandGestures

from utils import FeatxValEncoder, ChxValEncoder, FChCombxValEncoder, FeatxChxValEncoder, ChxFeatxValEncoder
encoders = {
    "fv":FeatxValEncoder,
    "cv": ChxValEncoder,
    "fChVal":FChCombxValEncoder,
    "featChVal":FeatxChxValEncoder,
    "chFeatVal":ChxFeatxValEncoder
}

def map_to_integers(tensor):
    # return (tensor*10000).to(torch.int)
    return(tensor*10000).to(torch.int)

def scaling(X, sigma=0.1):
    """ Data augmentation via randomly scaling with noise

        INPUTS:
        ------------
            X - (numpy array) ECG signal
            sigma - (float) threshold for random data augmentation

        OUTPUTS:
        ------------
            X * myNoise - (numpy array) randomly scaled signal
    """
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def shift(sig, interval=20):
    """ Data augmentation via randomly shifting signal

        INPUTS:
        ------------
            sig - (numpy array) ECG signal
            interval - (int) range of shift

        OUTPUTS:
        ------------
            sig - (numpy array) randomly shifted signal
    """

    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000
    return sig


def transform(sig, train=False):
    """ Tranformer function for data augmentation

        INPUTS:
        ------------
            sig - (numpy array) ECG signal
            train - (bool) data augmentation only in case of Training

        OUTPUTS:
        ------------
            sig - (numpy array) randomly augmented signal
    """

    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig

class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads, classes):
        """ Prepare and get the dataset

            INPUTS:
            ------------
                phase - (string) train, val or test?
                data_dir - (os path) path to data directory
                label_csv - ()

            OUTPUTS:
            ------------
        """
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir #.replace('\\', '\\\\')
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = classes
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        ecg_path = os.path.join(self.data_dir, patient_id)
        ecg_data, _ = wfdb.rdsamp(ecg_path)
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-SAMPLING:, self.use_leads] # last 30 s
        result = np.zeros((SAMPLING, self.nleads)) # 30 s, 500 Hz # 10 s, 100 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return map_to_integers(torch.from_numpy(result).float()), torch.argmax(torch.from_numpy(labels).float())

    def __len__(self):
        return len(self.labels)


def split_data(args=None, seed=42):
    """ Split data in train, val and test sets

        INPUTS:
        ------------
            seed - (int) seed for repeatable random split

        OUTPUTS:
        ------------
            folds[:8], folds[8:9], folds[9:] - (list) random splits
    """
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)

    train_folds = folds[:8]
    val_folds = folds[8:9]
    test_folds = folds[9:]

    return train_folds, val_folds, test_folds

def build_dataloader(train_folds, val_folds, test_folds, data_dir, label_csv, leads, classes, batch_size, num_workers):
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads, classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads, classes)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads, classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def experiment():
    train_folds, val_folds, test_folds = split_data()
    train_ld, val, test_ld = build_dataloader(train_folds, val_folds, test_folds, DATA_DIR, LABEL_CSV, "all", classes, 1, 1)

    encode = Encoder(DIMENSIONS, SAMPLING, 12, use_ngrams)
    encode = encode.to(device)

    num_classes = len(classes)
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)

    with torch.no_grad():
        
        for iter in range(EPOCH):
            for samples, targets in tqdm(train_ld, desc="Training"):
                samples = samples.to(device)
                targets = targets.to(device)

                sample_hv = encode(samples)
                if (ADD_ONLINE):
                    model.add_online(sample_hv, targets)
                else:
                    model.add(sample_hv, targets)
            print("Finished epoch - ", iter)
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        model.normalize()

        for samples, targets in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            accuracy.update(output.cpu(), targets)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


if __name__ == "__main__":
    # Get args
    args = parse_args()

    EPOCH = args.epoch
    DIMENSIONS = args.dimensions 
    NUM_LEVELS = args.num_levels
    BATCH_SIZE = args.batch_size
    N_GRAM_SIZE = args.n_gram
    MAX_VAL =  args.max_val
    MIN_VAL =  args.min_val
    SAMPLING = args.sampling
    SEED = args.seed
    LABEL_CSV = args.label_csv
    DATA_DIR = args.data_dir
    ADD_ONLINE = args.add_online
    Encoder = encoders[args.encoding]
    use_ngrams = args.use_ngram
    # classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    if args.classes=='all':
        classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        num_classes = len(classes)
    else:
        classes = args.classes.split(',')
        num_classes = len(classes)
    MAX = 5.1560
    MIN = -5.2297
    print("Starting Experiment with following configurations:")
    print("DIMENSIONS: ", DIMENSIONS)
    print("NUM_LEVELS: ", NUM_LEVELS)
    print("N_GRAM_SIZE: ", N_GRAM_SIZE)
    print("MAX_VAL: ", MAX_VAL)
    print("MIN_VAL: ", MIN_VAL)
    print("SAMPLING: ", SAMPLING)
    print("ENCODER: ", args.encoding)
    print("use_ngrams: ", use_ngrams)
    print("CLASSES: ", classes)
    
    
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        experiment()
    except Exception as e:
        print("Experiment Failed with - ", str(e))
        
    print("--------------------------------------------------------------")