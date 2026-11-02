import torch

def f(x):
    return x**2

x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

# 1. Analytical: derivative calculated by hand
analytical = 2 * x.detach()

# 2. Autograd: derivative calculated by PyTorch
f(x).backward()
autograd = x.grad

# 3. Numerical: estimate using two nearby inputs
epsilon = 1e-5
with torch.no_grad():
    numerical = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

print(f"Analytical: {analytical.item():.8f}")
print(f"Autograd:   {autograd.item():.8f}")
print(f"Numerical:  {numerical.item():.8f}")

assert torch.allclose(autograd, analytical)
assert torch.allclose(autograd, numerical)

print("All gradients agree!")