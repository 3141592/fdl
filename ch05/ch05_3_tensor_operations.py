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
print("x_t: ", x_t)
print("")

print("Tensor from torch.")
zeros_t = torch.zeros((2,3))
print("zeros_t: ", zeros_t)
ones_t = torch.ones((2,3))
print("ones_t: ", ones_t)
rand_t = torch.randn((2,3))
print("rand_t: ", rand_t)

print("")
print("Ch5. Tensor Attributes")
print("zeros_t.shape: ", zeros_t.shape)
print("zeros_t.dtype: ", zeros_t.dtype)
print("zeros_t.device: ", zeros_t.device)

print("")
print("Ch5. Tensor Operations")
print("rand_t*5: ", rand_t * 5)
print("(2,1,2) tensor: ", torch.ones((2,1,2)))
print("(2,2,2) tensor: ", torch.ones((2,2,2)))

x1_t = torch.tensor([[1,2],[3,4]])
x2_t = torch.tensor([[1,2,3],[4,5,6]])
print("x1_t: ", x1_t)
print("x2_t: ", x2_t)
print("torch.matmul(x1_t,x2_t): ", torch.matmul(x1_t,x2_t))
print("torch.matmul(x2_t,x1_t): ", torch.matmul(x2_t,x1_t))



