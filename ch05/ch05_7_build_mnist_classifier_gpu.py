import os
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# clear the screen
#os.system('clear')

# Check if CUDA is available
if torch.cuda.is_available():
    # CUDA is available
    print("CUDA is available.")
    device = torch.device('cuda')  # Specify GPU device
else:
    # CUDA is not available, fall back to CPU
    print("CUDA is not available.")
    device = torch.device('cpu')

print("")
from torchvision.datasets import MNIST
print("---------- ---------- ---------- ----------")
print("Ch5. Building the MNIST Classifier in PyTorch")

class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)

def find_tensors_not_on_gpu(obj):
    for name, param in obj.named_parameters():
        if param.device != device:
            print(f"Parameter '{name}' is not on GPU")
    for name, buf in obj.named_buffers():
        if buf.device != device:
            print(f"Buffer '{name}' is not on GPU")

def move_params_to_gpu(model):
    print(device)
    for param in model.parameters():
        param.data = param.data.to(device)

def move_params_by_name_to_gpu(model, param_names):
    for name, param in model.named_parameters():
        print(device)
        param.data = param.data.to(device)
        if name in param_names:
            param.data = param.data.to(device)
        else:
            print("Name not found")

print("")
print("Load in MNIST dataset from PyTorch")

train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

print("")
print("Instantiate model, optimizer, and hyperparameter(s)")
in_dim, feature_dim, out_dim = 784, 256, 10
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()
epochs = 40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
classifier.to(device)

# Check tensors not on GPU in the model
find_tensors_not_on_gpu(classifier)
move_params_to_gpu(classifier)
param_names_to_move = ['classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias']
move_params_by_name_to_gpu(classifier, param_names_to_move)
find_tensors_not_on_gpu(classifier)

optimizer = optim.SGD(classifier.parameters(), lr=lr)

print(classifier)

def train(classifier=classifier,
          optimizer=optimizer,
          epochs=epochs,
          loss_fn=loss_fn):

    classifier.train()
    loss_lt = []
    for epoch in range(epochs):
        running_loss = 0.0
        
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print("Keep track of sum of loss of each minibatch")
            running_loss += computed_loss.item()
        loss_lt.append(running_loss/len(train_loader))
        print("Epoch: {}, Training loss: {}".format(epoch+1,
            running_loss/len(train_loader)))

    plt.plot([i for i in range(1,epochs+1)], loss_lt)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
    plt.show()

    print("Save state to file as checkpoint")
    torch.save(classifier.state_dict(), 'mnist.pt')

def test(classifier=classifier,
         loss_fn=loss_fn):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)
            #print("Get loss and accuracy")
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)

        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item()/len(test_loader)*64,
            accuracy*100.0/(len(test_loader)*64)))

train(classifier, optimizer, epochs, loss_fn)

test(classifier, loss_fn)


