#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import openhd as hd


# In[2]:


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
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




# In[4]:


# Q = 10
D = 10000
hd.init(D=D, context=globals())


# In[ ]:





# In[5]:


@hd.run
def create_random_bases():
    position_base = hd.draw_random_hypervector()
    intensity_base = hd.draw_random_hypervector()
    return position_base, intensity_base


# In[6]:


@hd.run
def create_position_intensity_hvs(n_position, n_intensity, position_base, intensity_base):
    position_hvs = hd.hypermatrix(n_position)
    for i in range(n_position):
        position_hvs[i] = hd.permute(position_base, i)
 
    intensity_hvs = hd.hypermatrix(n_intensity)
    for i in range(n_intensity):
        intensity_hvs[i] = hd.permute(intensity_base, i)
    
    return position_hvs, intensity_hvs


# In[7]:


with hd.utils.timing("Base hypervectors"):
    position_base, intensity_base = create_random_bases()
#     image_hv = hd.hypervector()


# In[8]:


with hd.utils.timing("Feature hypervectors"):
    position_hvs, intensity_hvs = create_position_intensity_hvs(784, 255, position_base, intensity_base)
#     image_hv = hd.hypervector()


# In[9]:


position_hvs.to_numpy().shape, intensity_hvs.to_numpy().shape


# In[10]:


def encode_pixels(flattened_image, output_hypervector,
                  position_hvs, intensity_hvs, n_position = 784, n_intensity = 255): # arguments passed by args
    for pixel_idx in range(784):
        output_hypervector += position_hvs[pixel_idx] * intensity_hvs[flattened_image[pixel_idx]]
        
        


# In[11]:


def bipolarize(arr):
    result = np.where(arr < 0, -1, np.where(arr > 0, 1, np.random.choice([-1, 1])))
    return result


# In[19]:


def validate(labels, pred_labels):
    n_correct = (pred_labels == labels).sum()
    n_labels = len(labels)
#     print(n_correct, n_labels, n_correct / float(n_labels) * 100)
    return  n_correct / float(n_labels) * 100


# # Epoch Based Training

# In[13]:


@hd.run
def retrain(class_hvs, hv_matrix, labels, N):
    for idx in range(N): # Iterate through each image
        class_hvs[labels[idx]] += hv_matrix[idx]
    return class_hvs


# In[14]:


BATCH_SIZE = 9876
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=1)
test_loader = DataLoader(test_data, 
                         batch_size=50,
                         shuffle=True, 
                         num_workers=1)


# In[22]:


EPOCHS = 2000
n_classes = 10
class_hvs = hd.hypermatrix(n_classes)
train_performances = []
for e in range(EPOCHS):
    # Fetch Current Image batch
    images, labels = next(iter(train_loader))
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2]*images.shape[3]) * 255
    labels = np.array(labels, dtype = np.int32)
    # Encode the images of this batch
    hv_matrix = hd.encode(
            encode_pixels, extra_args = (position_hvs, intensity_hvs, 784, 255),
            feature_matrix = images
            )
    # bipolarize
    hv_numpy = hv_matrix.to_numpy()
    hv_numpy = bipolarize(hv_numpy)
    hv_matrix = hv_matrix.from_numpy(hv_numpy)
    # add to class_hvs
    class_hvs = retrain(class_hvs, hv_matrix, labels, BATCH_SIZE)
    # bipolarize
    class_hvs_np = class_hvs.to_numpy()
    class_hvs_np = bipolarize(class_hvs_np)
    class_hvs = class_hvs.from_numpy(class_hvs_np)
    v = validate(labels, hd.search(class_hvs, hv_matrix).to_numpy())
    train_performances.append(v)
    print("At epoch ",e, ": ", v)
    


# # Testing

# In[23]:


images_tst, labels_tst = next(iter(test_loader))
images_tst = images_tst.reshape(images_tst.shape[0], images_tst.shape[1]*images_tst.shape[2]*images_tst.shape[3]) * 255
labels_tst = np.array(labels_tst, dtype = np.int32)


# In[24]:


hv_matrix_tst = hd.encode(
        encode_pixels, extra_args = (position_hvs, intensity_hvs, 784, 255),
        feature_matrix = images_tst
        )
hv_numpy_tst = hv_matrix_tst.to_numpy()
hv_numpy_tst = bipolarize(hv_numpy_tst)
hv_matrix_tst = hv_matrix_tst.from_numpy(hv_numpy_tst)

# In[25]:

print("On Test Data", validate(labels_tst, hd.search(class_hvs, hv_matrix_tst).to_numpy()))

print("On Train Data", validate(labels, hd.search(class_hvs, hv_matrix).to_numpy()))

# In[33]:


import matplotlib.pyplot as plt
plt.plot(np.arange(len(train_performances)), train_performances)
plt.savefig("Training_performances.png")


# # Guided Training
# ### Only update the class vectors which are wrong

# In[ ]:




