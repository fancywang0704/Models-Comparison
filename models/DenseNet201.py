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
import torch.nn.functional as F

batch_size = 1 #1，2，4，8，16，32，64

# ----- Image data changes
transform_train = transforms.Compose([
    #transforms.RandomHorizontalFilp(),
    transforms.Resize(size=96),
    #transforms.RdandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(size=96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/my_demo/cifar10_data_train', train=True,
                                        download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/my_demo/cifar10_data_test', train=False,
                                       download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# ----- DenseNet201
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer
 
class dense_block(nn.Module):
    # growth_rate即output_channel
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(
                conv_block(in_channel=channel, out_channel=growth_rate)
            )
            channel += growth_rate
            self.net = nn.Sequential(*block)
 
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x
 
 
blk = dense_block(in_channel=3, growth_rate=10, num_layers=4)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
#print(Y.shape) # torch.Size([4, 43, 8, 8])
 
 
def transition_block(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer
 
 
blk = transition_block(in_channel=43, out_channel=10)
#print(blk(Y).shape) # torch.Size([4, 10, 4, 4])
 
 
class DenseNet(nn.Module):
    def __init__(self, in_channel, num_classes=10, growth_rate=32, block_layers=[6, 12, 48, 32]):     #DenseNet201
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
 
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition_block(channels, channels // 2)) 
                channels = channels // 2
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.classifier = nn.Linear(channels, num_classes)
 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
net = DenseNet(in_channel=3, num_classes=10)

model = net.to(device)
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
torch.save(model.state_dict(), "densenet201.pth")
print("Saved PyTorch Model State to densenet201.pth")


