# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import helper
from collections import OrderedDict
import PIL


def test():
    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    epochs = 3
    #
    criterion = nn.CrossEntropyLoss()
    # Build a feed-forward network
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('logits', nn.Linear(hidden_sizes[1], output_size))]))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Forward pass through the network and display output
    steps = 0
    print_every=40
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            images.resize_(images.size()[0], 784)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))
                running_loss = 0
    torch.save(model, 'first.pth')
    # helper.view_classify(images[0].view(1, 28, 28), ps)


def load_data():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize(0.5, 0.5),
                                    ])
    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data2/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('MNIST_data2/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader


def load_data2():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5),
                                    ])
    # Download and load the training data
    trainset = datasets.FashionMNIST('fashion_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST('fashion_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader

def test2():
    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    epochs = 3
    #
    criterion = nn.CrossEntropyLoss()
    # Build a feed-forward network
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('logits', nn.Linear(hidden_sizes[1], output_size))]))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Forward pass through the network and display output
    steps = 0
    print_every = 40
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            images.resize_(images.size()[0], 784)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))
                running_loss = 0
    torch.save(model, 'fashion.pth')
    # helper.view_classify(images[0].view(1, 28, 28), ps)

def test_grad():
    x= torch.randn(2,2,2,requires_grad=True)
    y= x**2
    print(x)
    print(y)
    z = y.mean()
    print(z)
    z.backward()
    print(x.grad)


def check():
    model = torch.load('first.pth')
    images, labels = next(iter(testloader))
    img = images[0].view(1, 784)
    logits = model.forward(img)
    ps = F.softmax(logits, dim=1)
    helper.view_classify(img.view(1, 28, 28), ps)


if __name__ == '__main__':
    # trainloader, testloader = load_data2()
    # # test()
    # images, label = next(iter(trainloader))
    # helper.imshow(images[0, :])
    img = PIL.Image.open("/Users/zyl_home/Desktop/1.jpeg")
    img_re = img.resize((244,244))
    print(np.array(img_re).shape)


