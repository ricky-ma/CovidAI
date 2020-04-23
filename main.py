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

    batch_size = 7
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


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    predictions = []

    with torch.no_grad():
        for i, (inputs) in enumerate(data_loaders['test']):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds)
            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')
            #     ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            #     imshow(inputs.cpu().data[j])
            #
            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
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
    # device = "cpu"

    # # Get a batch of training data
    # inputs, classes = next(iter(data_loaders['train']))
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    # time.sleep(60)


    model_ft = models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)

    visualize_model(model_ft)


    # model = Net()
    # model.to(device)
    #
    # # training
    # start_time = time.time()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    # for epoch in range(10):  # loop over the dataset multiple times
    #     running_loss = 0.0
    #     for i, data in enumerate(data_loaders['train'], 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 5 == 4:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 5))
    #             running_loss = 0.0
    # print('Finished Training')
    # print('Trained for: ' + str(time.time()-start_time))
    #
    # # prediction on validations
    # dataiter = iter(data_loaders['validation'])
    # images, labels = dataiter.next()
    #
    # correct = 0
    # total = 0
    # predictions = []
    # with torch.no_grad():
    #     for data in data_loaders['validation']:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         predictions.append(predicted)
    # print(predictions)
    # print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
    #
    # # prediction on test
    # model.eval()
    # dataiter = iter(data_loaders['test'])
    # images = dataiter.next()
    #
    # predictions = []
    # with torch.no_grad():
    #     for data in data_loaders['test']:
    #         images = data.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         predictions.append(predicted)
    # print(predictions)






