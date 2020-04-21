# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision
import time
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import numpy as np
from model import build_model, compile_model, augment_data, fit


class CovidDatasetTrain(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx]


def make_data_loaders():
    train_dataset = CovidDatasetTrain(train_imgs, train_labels)
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=6),
        "test": DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=6),
    }


if __name__ == '__main__':
    plt.ion()  # interactive mode

    train_imgs = pk.load(open("data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pk.load(open("data/train_labels_512.pk", 'rb'), encoding='bytes')
    test_imgs = pk.load(open("data/test_images_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders()
    dataset_sizes = {'train': len(data_loaders['train'].dataset), 'test':len(data_loaders['test'].dataset)}

    class_names = ['covid', 'background']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train, y_train = data_loaders['train']
    x_test, y_test = data_loaders['test']
    # x_train, y_train = x_train.to(device), y_train.to(device)
    # x_test, y_test = x_test.to(device), y_test.to(device)
    #
    # model = build_model(x_train)
    # compile_model(model)
    # datagen = augment_data()
    # history = fit(model, x_train, y_train, x_test, y_test, datagen)
    #
    # plt.plot(history.history['acc'], label='training accuracy')
    # plt.plot(history.history['val_acc'], label='testing accuracy')
    # plt.title('Accuracy')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.legend()
    #
    # plt.plot(history.history['loss'], label='training loss')
    # plt.plot(history.history['val_loss'], label='testing loss')
    # plt.title('Loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()

    # max_epochs = 100
    # for epoch in range(max_epochs):
    #     for local_batch, local_labels in data_loaders['train']:
    #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    #
    #         model = build_model(local_batch)
    #         compile_model(model)
    #         datagen = augment_data()
    #         fit(model, local_batch, local_labels,)




