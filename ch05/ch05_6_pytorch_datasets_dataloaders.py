import os
import torch
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
        img_pth = os.path.join(self.img_dir, "cat.{}.jpg".format(idx))
        #img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
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


