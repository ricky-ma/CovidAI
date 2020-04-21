# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
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
    random_seed = 39

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


if __name__ == '__main__':
    plt.ion()  # interactive mode

    train_imgs = pk.load(open("data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pk.load(open("data/train_labels_512.pk", 'rb'), encoding='bytes')
    test_imgs = pk.load(open("data/test_images_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders()
    dataset_sizes = {'train': len(data_loaders['train'].dataset), 'test':len(data_loaders['test'].dataset)}
    print(dataset_sizes)

    class_names = ['covid', 'background']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model.to(device)

    # training
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
    print('Finished Training')
    print('Trained for: ' + str(time.time()-start_time))

    # prediction on validations
    dataiter = iter(data_loaders['validation'])
    images, labels = dataiter.next()

    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in data_loaders['validation']:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.append(predicted)
    print(predictions)
    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))

    # prediction on test
    model.eval()
    dataiter = iter(data_loaders['test'])
    images = dataiter.next()

    predictions = []
    with torch.no_grad():
        for data in data_loaders['test']:
            images = data.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
    print(predictions)
    # print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))






