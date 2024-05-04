import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import openhd as hd

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = False,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)



from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, 
                         batch_size=60000, 
                         shuffle=True, 
                         num_workers=1)

feature_matrix, labels = next(iter(train_loader))

feature_matrix = feature_matrix.reshape((60000, 28*28))


Q = 10
D = 10000
hd.init(D=D, context=globals())

N = feature_matrix.shape[0]
F = feature_matrix.shape[1]



@hd.run
def create_random_bases():
    id_base = hd.draw_random_hypervector()
    level_base = hd.draw_random_hypervector()
    return id_base, level_base


@hd.run
def create_ids(F, id_base):
    id_hvs = hd.hypermatrix(F) # np.zeros(F, N) (not the empty list) 
    for f in range(F):
        id_hvs[f] = hd.permute(id_base, f)

    return id_hvs

@hd.run
def create_levels(Q, level_base):
    level_hvs = hd.hypermatrix(Q+1) # np.zeros((Q+1), N) (not the empty list)
    for q in range(Q+1):
        idx = int(q/float(Q) * D) / 2
        level_hvs[q] = hd.flip(level_base, idx)
        level_hvs[q] = hd.shuffle(level_hvs[q], 0)

    return level_hvs



with hd.utils.timing("Base hypervectors"):
    id_base, level_base = create_random_bases()
    id_hvs = create_ids(F, id_base)
    level_hvs = create_levels(Q, level_base)


print(id_hvs.to_numpy().shape)