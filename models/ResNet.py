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
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
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

# ----- ResNet
net = models.resnet18(pretrained=True)
#net = models.resnet34(pretrained=True)
net.avgpool = nn.AvgPool2d(kernel_size=1,stride=1)
net.fc = nn.Linear(512,10)

#net = models.resnet50(pretrained=True)
#net = models.resnet101(pretrained=True)
#net = models.resnet152(pretrained=True)
#net.avgpool = nn.AvgPool2d(kernel_size=1,stride=1)
#net.fc = nn.Linear(2048,10)

model = net.to(device).eval()
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
    train(train_dataloader, model, loss_fn, optimizer)
    startTime = timeit.default_timer() 
    evaluteTop1(test_dataloader, model)
    evaluteTop5(test_dataloader, model)
    stopTime = timeit.default_timer() 
    print('Running time: %5.1fs.'%(stopTime - startTime))
print("Done!")

#Save
torch.save(model.state_dict(), "resnet18.pth")
print("Saved PyTorch Model State to resnet18.pth")

