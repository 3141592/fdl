import os
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchinfo import summary


print("")
print("---------- ---------- ---------- ----------")
print("Ch7. Convolutional Neural Networks")

print("")
print("---------- ---------- ---------- ----------")
print("Full Description of the Convolutional Layer")
layer = nn.Conv2d(in_channels = 3,
                  out_channels = 64,
                  kernel_size = (5, 5),
                  stride = 2,
                  padding = 1
                 )

layer.summary()
