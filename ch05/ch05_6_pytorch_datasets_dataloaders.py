import os
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

# clear the screen
os.system('clear')

print("")
print("---------- ---------- ---------- ----------")
print("Ch5. PyTorch Datasets and Dataloaders")

class ImageDataset(data.Dataset):
    def __init__(self, img_dir, label_file):
        super(ImageDataset, self).__init__()
        self.img_dir = img_dir
        self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
        self.transforms = transforms.ToTensor()

    def __getitem__(self, idx):
        img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
        img = Image.open(img_pth)
        img = self.transforms(img).flatten()
        #print("img: ")
        #print(img)
        label = self.labels[idx]
        return {"data":img, "label":label}

    def __len__(self):
        return len(self.labels)

train_dataset = ImageDataset(img_dir='./data/train/',
                             label_file='./data/train/labels.npy')

train_loader = data.DataLoader(train_dataset,
                          batch_size=4,
                          shuffle=True)

print("---------- ---------- ---------- ----------")
print("Iterate through the dataloader")
for minibatch in train_loader:
    data, labels = minibatch['data'], minibatch['label']
    print("data.shape: ", data.shape)
    print("data:")
    print(data)
    print("labels.shape: ", labels.shape)
    print("labels:")
    print(labels)

print("---------- ---------- ---------- ----------")
print("The dataloader also does the work of stacking all of the examples into a single tensor the can be run through the network")
class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
        self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        out = self.layer2(x)
        return out

print("")
print("---------- ---------- ---------- ----------")
print("Generate the output of a model on some input")
no_examples = 10
in_dim, feature_dim, out_dim = 784, 256, 10

classifier = BaseClassifier(in_dim, feature_dim, out_dim)

for minibatch in train_loader:
    data, labels = minibatch['data'], minibatch['label']
    out = classifier(data)
    print("out: ", out)


















