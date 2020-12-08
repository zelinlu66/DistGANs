#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:43:10 2020

@author: 7ml
"""


import numpy as np

# load data
from torchvision import datasets
import torch.utils.data as data


# load the training data
train_data = datasets.CIFAR100('./cifar100_data', train=True, download=True)
# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate(
    [np.asarray(train_data[i][0]) for i in range(len(train_data))]
)
# print(x)
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean_global = np.mean(x, axis=(0, 1))
train_var_global = np.var(x, axis=(0, 1))
# the the mean and std
print(train_mean_global, train_var_global)


from tensorflow.keras import datasets
import numpy as np

(
    (train_images, train_labels),
    (test_images, test_labels),
) = datasets.cifar100.load_data()
print('Images Shape: {}'.format(train_images.shape))
print('Labels Shape: {}'.format(train_labels.shape))

train_mean_labels = []

for i in range(0, 10):

    print("######################")
    print("Class: " + str(i))
    idx = (train_labels == i).reshape(train_images.shape[0])
    filtered_images = train_images[idx]
    print('Images Shape: {}'.format(filtered_images.shape))
    x = np.concatenate(
        [np.asarray(filtered_images[i]) for i in range(len(filtered_images))]
    )
    train_mean = np.mean(x, axis=(0, 1))
    train_mean_labels.append(train_mean)
    train_var = np.var(x, axis=(0, 1))
    # the the mean and std
    print(train_mean, train_var)
    print("######################")

print("Variance between groups: ", np.var(train_mean_labels))

g = 100
n_single_class = train_images.shape[0] / 100
n_tot = train_images.shape[0]

SS_b = np.var(train_mean_labels, axis=0) * (g - 1) * (n_single_class - 1)

SS_res = train_var_global * (n_tot - 1) - SS_b

F_ratio = (SS_b / (g - 1)) / (SS_res / (n_tot - g))

print("F ratio: ", F_ratio)
