import torch
import os

# clear the screen
os.system('clear')

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch.cuda.is_available(): ", torch.cuda.is_available())
print("device: ", device)

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
# ERROR print("torch.matmul(x2_t,x1_t): ", torch.matmul(x2_t,x1_t))

print("")
x3_t = torch.tensor([[[3,7,9],[2,4,5]],[[8,6,2],[3,9,1]]])
print("x3_t: ", x3_t)

print("")
i,j,k = 0,1,1
print("x3_t[i,j,k]: ", x3_t[i,j,k])

print("")
print("x3_t[0]: ", x3_t[0])
print("x3_t[0,:,:]: ", x3_t[0,:,:])

print("")
print("Use the : symbol to subset the data further:")
print("x3_t[0,1:3,:]: ", x3_t[0,1:3,:])

print("")
print("Set indices to new value:")
print("x3_t[0,1,2] = 1")
x3_t[0,1,2] = 1
print("x3_t: ", x3_t)

print("")
print("Set larger slice to new values:")
x_t = torch.randn(2,3,4)
print("x_t(2,3,4): ", x_t)
sub_tensor = torch.randn(2,4)
print("sub_tensor(2,4): ", sub_tensor)
x_t[0,1:3,:] = sub_tensor
print("x_t[0,1:3,:] = subtensor: ", x_t)

print("")
print("Gradients in PyTorch")
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.5, requires_grad=True)
print("x: ", x)
print("y: ", y)
print("z: ", z)
f = x**2 + y**2 + z**2
print("f = x**2 + y**2 + z**2")
print("f: ", f)
f.backward()
print("x.grad, y.grad, z.grad: ", x.grad, y.grad, z.grad)

print("")
g = x**2 + y**3 + z**4
print("g = x**2 + y**2 + z**2")
print("g: ", g)
g.backward()
print("x.grad, y.grad, z.grad: ", x.grad, y.grad, z.grad)














