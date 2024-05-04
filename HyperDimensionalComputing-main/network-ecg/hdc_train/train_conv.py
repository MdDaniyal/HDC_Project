#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path as pl

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wfdb
import torch.optim as optim


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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
        ecg_data = ecg_data[-15000:, self.use_leads] # last 30 s
        result = np.zeros((15000, self.nleads)) # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)


# In[6]:


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



# In[7]:


import torch
import torch.nn as nn

class THREE_CONVs(nn.Module):
    def __init__(self):
        super(THREE_CONVs, self).__init__()
        
        self.conv0 = nn.Conv1d(in_channels=12, out_channels=8, kernel_size=15, stride=2, padding=7)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=940, stride=940, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 9)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# In[8]:


train_folds, val_folds, test_folds = split_data()


# In[9]:


label_csv = "/home/hpc/iwi3/iwi3083h/network-ecg/train/labels/labels.csv"


# In[10]:


data_dir = "/home/hpc/iwi3/iwi3083h/data/CPSC"


# In[11]:


classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']


# In[12]:


train, val, test = build_dataloader(train_folds, val_folds, test_folds, data_dir, label_csv, "all", classes, 32, 1)


# In[13]:


# Instantiate the model
model = THREE_CONVs()

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
save_every = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # Training
    model.train()
    itr=0
    for inputs, labels in train:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/ Iteration: {itr}, Loss: {loss.item()}')
        itr+=1

    # Save the model every second epoch
    if (epoch + 1) % save_every == 0:
        torch.save(model.state_dict(), f'three_conv_{epoch + 1}.pth')

    # Validation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in val:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(torch.sigmoid(outputs), 1)
            _, label_classes = torch.max(labels, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == label_classes).sum().item()

        accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%')

# Testing
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for inputs, labels in test:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(torch.sigmoid(outputs), 1)
        _, label_classes = torch.max(labels, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == label_classes).sum().item()

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


# In[ ]:




