# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:22:47 2020

@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 
"""


from torchvision import transforms, datasets
import os
import torch


def mnist_data(rand_rotation=False, max_degree=90):
    if rand_rotation == True:
        compose = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.RandomRotation(max_degree),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    else:
        compose = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    out_dir = '{}/dataset'.format(os.getcwd())
    return datasets.MNIST(
        root=out_dir, train=True, transform=compose, download=True
    )


def cifar10_data():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    #out_dir = '{}/dataset'.format(os.getcwd())
    return datasets.CIFAR10(
        root='/home/zelinlu/final/DistGANs/dataset/cifar10', train=True, transform=compose, download=False
    )


def cifar100_data():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    out_dir = '{}/dataset'.format(os.getcwd())
    return datasets.CIFAR100(
        root=out_dir, train=True, transform=compose, download=True
    )


def imagenet_data():
    data_path = '/Users/7ml/Documents/ImageNet1k/'

    traindir = os.path.join(data_path, "train")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    return train_dataset
