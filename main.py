# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, models, transforms
import time
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import copy
from PIL import Image
from model import Net


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

    batch_size = 5
    validation_split = 0.1
    random_seed = 43

    # Creating data indices for training and validation splits:
    train_size = len(train_dataset)
    indices = list(range(train_size))
    split = int(np.floor(validation_split * train_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=6, sampler=train_sampler),
        "validation": DataLoader(train_dataset, batch_size=batch_size, num_workers=6, sampler=valid_sampler),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6),
    }


def fit(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('Predictions:')
            print(preds)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict(model):
    was_training = model.training
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (inputs) in enumerate(data_loaders['test']):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds)
        model.train(mode=was_training)
    print(predictions)


if __name__ == '__main__':
    plt.ion()  # interactive mode

    train_imgs = pk.load(open("data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pk.load(open("data/train_labels_512.pk", 'rb'), encoding='bytes')
    test_imgs = pk.load(open("data/test_images_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders()
    dataset_sizes = {'train': 63,
                     'validation': 7,
                     'test': len(data_loaders['test'].dataset)}
    print(dataset_sizes)

    class_names = ['covid', 'background']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ResNet18
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    # model_ft = fit(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    # predict(model_ft)


    # model_conv = torchvision.models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    # model_conv = model_conv.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #
    # model_conv = fit(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    # predict(model_conv)

    # DenseNet161
    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = fit(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    predict(model_ft)





