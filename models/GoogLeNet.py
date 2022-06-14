import sys, os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,  transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import timeit
import pandas as pd
from torch.autograd import Variable
import numpy as np
from torch import nn,optim
from torch.autograd import Variable


batch_size = 1 #1, 2, 4, 8, 16, 32, 64

# ----- Image data changes
transform_train = transforms.Compose([
    #transforms.RandomHorizontalFilp(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/my_demo/cifar10_data_train', train=True,
                                        download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/my_demo/cifar10_data_test', train=False,
                                       download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#GoogLeNet
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1*1卷积
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes,n1x1,kernel_size=1),
                                   nn.BatchNorm2d(n1x1),nn.ReLU(True))

        # 1*1+3*3
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes,n3x3red,kernel_size=1),
                                   nn.BatchNorm2d(n3x3red),nn.ReLU(True),
                                   nn.Conv2d(n3x3red,n3x3,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n3x3),nn.ReLU(True))

        #1*1+2个3*3
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes,n5x5red,kernel_size=1),
                                   nn.BatchNorm2d(n5x5red),nn.ReLU(True),
                                   nn.Conv2d(n5x5red,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True),
                                   nn.Conv2d(n5x5,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True))

        # 3*3池化+1*1
        self.conv4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                   nn.Conv2d(in_planes,pool_planes,kernel_size=1),
                                   nn.BatchNorm2d(pool_planes),nn.ReLU(True))
         
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        return torch.cat([out1, out2, out3, out4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, out_dim):
        super(GoogLeNet, self).__init__()
        self.pre_layer = nn.Sequential(nn.Conv2d(3,192,kernel_size=3,padding=1),
                                       nn.BatchNorm2d(192),
                                       nn.ReLU(True))

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = GoogLeNet(out_dim=10).to(device)
#print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-5)


# ----- Train
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        correct = 0
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], Accuracy: {(100*correct):>0.1f}% ")

# ----- Test
def evaluteTop1(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Top1 Error: \n Accuracy: {(100*correct):>0.1f}% \n")

def evaluteTop5(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    correct /= size
    print(f"Top5 Error: \n Accuracy: {(100*correct):>0.1f}% \n")


# ----- Epochs
epochs = 100 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #train(train_dataloader, model, loss_fn, optimizer)
    startTime = timeit.default_timer() 
    evaluteTop1(test_dataloader, model)
    evaluteTop5(test_dataloader, model)
    stopTime = timeit.default_timer() 
    print('Running time: %5.1fs.'%(stopTime - startTime))
print("Done!")

#Save
torch.save(model.state_dict(), "Googlenet.pth")
print("Saved PyTorch Model State to Googlenet.pth")
