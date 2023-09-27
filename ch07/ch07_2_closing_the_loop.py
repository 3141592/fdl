import os
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchinfo import summary
import torchsummary

print("")
print("Load in MNIST dataset from PyTorch")

train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("")
print("Closing the loop on MNIST with Convolution Networks")

device = "cpu"

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.fc1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(7*7*64, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 10)
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc1(x)

lr = 1e-4
num_epochs = 40

model = MNISTConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print("Model moved to GPU")
else:
    print("GPU is not available, using CPU instead")

print("")
print("Train our network using the Adam optimizer.")
for epochs in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for inputs, labels in train_loader:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputs = inputs.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        _, idx = outputs.max(dim=1)
        num_correct += (idx == labels).sum().item()
    print('epochs: {} loss: {} Accuracy: {}'.format(epochs, running_loss/len(train_loader),
        num_correct/len(train_loader)))


