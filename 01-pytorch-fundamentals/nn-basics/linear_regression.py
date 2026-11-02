import torch

# Training data
X = torch.tensor([1., 2., 3., 4.])
y = torch.tensor([3., 5., 7., 9.])

# Parameters we want to learn
w = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.01

for epoch in range(501):
    # 1. Predict
    prediction = w * X + b

    # 2. Calculate mean squared error
    loss = ((prediction - y) ** 2).mean()

    # 3. Calculate gradients
    loss.backward()

    # 4. Update parameters without tracking the update
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 5. Clear gradients before the next iteration
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print(f"\nLearned: y = {w.item():.3f}x + {b.item():.3f}")
print(f"Prediction for x=5: {(w * 5 + b).item():.3f}")