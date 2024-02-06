import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

# Step 1: Load the MNIST dataset with normalization transform
transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL images to PyTorch tensors
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Step 2: Convert the dataset to a PyTorch tensor
mnist_data = mnist_dataset.data  # This is a tensor of images
mnist_data = mnist_data.float()  # Convert to float tensor

# Step 3: Calculate mean and standard deviation
mean = torch.mean(mnist_data / 255.0)  # Normalize pixel values to [0, 1] range and calculate mean
std = torch.std(mnist_data / 255.0)   # Calculate standard deviation

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

first_image = mnist_data[5]

transform = transforms.ToPILImage()
img = transform(first_image)
img.show()
