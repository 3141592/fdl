import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

# Step 1: Load the MNIST dataset with normalization transform
transform = transforms.Compose([
    #transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0,
                           contrast=0,
                           saturation=0,
                           hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.1307,),
                         std = (0.3081,)
                         )
    ])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#print("mnist_dataset: ", type(mnist_dataset))
#print("mnist_dataset.data: ", type(mnist_dataset.data))

# Step 2: Convert the dataset to a PyTorch tensor
mnist_data = mnist_dataset.data  # This is a tensor of images

images, labels = next(iter(mnist_dataset))
print(images.shape)

first_image = mnist_data[5]
#print("first_image: ", type(first_image))

#print(type(mnist_dataset.data))
transform = transforms.ToPILImage()
img = transform(first_image)
img.show()
#save_image(first_image, 'first_mnist_image.png')
