# -*- coding: UTF-8 -*-
from logTime import log_time
import torch
from torchvision import datasets, transforms
import torchvision
import sys
# plt.rc("font",family="SimHei",size="12")


@log_time
def test():
    x = torch.randn(2,2, requires_grad=True)
    print(x)
    y = x ** 2
    print(y)
    print(y.grad_fn)


def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    load_data()
    # print(sys.path)
