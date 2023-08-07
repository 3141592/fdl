import torch

print("Ch5. PyTorch Tnesors")
print("Ch5. Tensor Init")

print("Tensor from array.")
arr = [1,2]
tensor = torch.tensor(arr)
val = 2.0
tensor = torch.tensor(val)
print(tensor)
print("")

print("Tensor from numpy array.")
import numpy as np
np_arr = np.array([1,2])
x_t = torch.from_numpy(np_arr)
print(x_t)
print("")

print("Tensor from torch.")
zeros_t = torch.zeros((2,3))
print(zeros_t)
ones_t = torch.ones((2,3))
print(ones_t)
rand_t = torch.randn((2,3))
print(rand_t)

print("")
print("Ch5. Tensor Attributes")
print("zeros_t.shape: ", zeros_t.shape)
print("zeros_t.dtype: ", zeros_t.dtype)
print("zeros_t.device: ", zeros_t.device)





