import torch
import pytest
#------- Broadcasting ---------
# Broadcasting lets an operation combine tensors whose shapes differ.

x= torch.ones(3,4)
b = torch.tensor([1,2,3,4])

print(x)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
print("Shape of x", x.shape) # (3,4)
print(b)
# tensor([1, 2, 3, 4])
print("Shape of b", b.shape) # (4,)

y= x+b
print(y)
# tensor([[2., 3., 4., 5.],
#         [2., 3., 4., 5.],
#         [2., 3., 4., 5.]])
print("Shape of y", y.shape) # (3,4)

# Common trap: a tensor of shape (3,) cannot broadcast with (3, 4), because the trailing dimensions are 3 and 4. For a value per row, use (3, 1)


# -------------------- Strides ---------------------------
# A tensor has both a logical shape and a memory layout.
# Its strides say how far to move through storage, measured in elements, when an index increases by one.

# x = torch.arange(12).reshape(3, 4)

# x.shape     # (3, 4)
# x.stride()  # (4, 1)

# t = x.T

# t.shape     # (4, 3)
# t.stride()  # (1, 4)

# A tensor is contiguous when its elements follow the expected dense layout for its shape. A transpose is typically noncontiguous.

#---------------- Chain Rule -----------------
# u = 10x + 5
# L = u^2

# dl/dx = dl/du * du/dx = 20u = 20(10*x +5) = 1100
new_x = torch.tensor(5., requires_grad= True)

u = 10* new_x + 5
L = u**2

L.backward()
print(x.grad) # tensor(1100.)


x = torch.tensor([2., 3.], requires_grad=True)
y = x**2

y.backward(torch.tensor([1., 1.]))
print(x.grad)  # tensor([4., 6.])
