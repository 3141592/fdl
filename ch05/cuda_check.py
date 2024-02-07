import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    # CUDA is available
    device = torch.device('cuda')  # Specify GPU device
else:
    print("CUDA is not available")
    # CUDA is not available, fall back to CPU
    device = torch.device('cpu')

# Create a tensor
x = torch.randn(3, 3)

# Move tensor to the specified device (GPU or CPU)
x = x.to(device)

# Create a model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(3, 1)  # Example linear layer

    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = SimpleModel()

# Move the model to the specified device (GPU or CPU)
model = model.to(device)

print(model)
