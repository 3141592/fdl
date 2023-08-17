import torch
import os
import torch.nn as nn

# clear the screen
#os.system('clear')

print("Ch5. PyTorch Tnesors")
print("The PyTorch nn Module")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)

print("")
print("---------- ---------- ---------- ----------")
print("Initialize a weight matrix for a feed-forward neural network:")
in_dim, out_dim = 32, 10
vec = torch.randn(32)
layer = nn.Linear(in_dim, out_dim, bias=True)
out = layer(vec)
print("vec: ", vec)
print("layer: ", layer)
print("out: ", out)

print("")
print("---------- ---------- ---------- ----------")
print("A feed-forward network as a composition of Layers:")
in_dim, feature_dim, out_dim = 784, 256, 10
vec = torch.randn(784)
layer1 = nn.Linear(in_dim, feature_dim, bias=True)
layer2 = nn.Linear(feature_dim, out_dim, bias=True)
out = layer2(layer1(vec))
print("layer1: ", layer1)
print("layer2: ", layer2)
print("out: ", out)

print("")
print("---------- ---------- ---------- ----------")
print("The nn module provides nonlinearities such as ReLU:")
relu = nn.ReLU()
out = layer2(relu(layer1(vec)))
print("relu: ", relu)
print("out: ", out)

print("")
print("---------- ---------- ---------- ----------")
print("Encapsulate the simple two-layer neural network:")
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
x = torch.randn((no_examples, in_dim))
x = x * 10
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier(x)

print("x: ", x)
print("classifier: ", classifier)
print("out: ", out)

print("")
print("---------- ---------- ---------- ----------")
print("Use the cross-entropy loss metric")
loss = nn.CrossEntropyLoss()
target = torch.tensor([0, 3, 2, 8, 2, 9, 3, 7, 1, 6])
computed_loss = loss(out, target)

print("loss: ", loss)
print("target: ", target)
print("computed_loss: ", computed_loss)
computed_loss.backward()
print("computed_loss.backward(): ", computed_loss)

print("")
print("---------- ---------- ---------- ----------")
print("View the shape of each parameter in the neural network")
for p in classifier.parameters():
    print("p.shape: ", p.shape)
    #print("p.requires_grad: ", p.requires_grad)
    #print("p.grad: ", p.grad)
    #print("p.grad_fn: ", p.grad_fn)

print("")
print("---------- ---------- ---------- ----------")
print("Define the optimizer")
from torch import optim

lr = 1e-1
optimizer = optim.SGD(classifier.parameters(), lr=lr)

print("")
print("optimizer:", optimizer)
optimizer.step()
optimizer.zero_grad()

print("loss: ", loss)
print("target: ", target)
print("computed_loss: ", computed_loss)
print("computed_loss.backward(): ", computed_loss)





