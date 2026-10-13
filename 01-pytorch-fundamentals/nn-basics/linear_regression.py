import torch
from torch import nn
from datasets import train_dataloader, test_dataloader

device= torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

images, labels = next(iter(train_dataloader))
print(f"Input shape {images.shape}")  # torch.Size([64, 1, 28, 28])
# That means 64 images, each with 1 channel 28 height 28 width.
input_size = images[0].numel()
print(f"Input size {input_size}")  # 784
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self,x):
        x= self.flatten(x)
        y = self.linear(x)
        return(y)

# ------------- Create a Model -----------------
model = LinearRegression().to(device)
print(model)

# --------------- Test a dataset -----------------
images, labels = next(iter(test_dataloader))

X = images[0:1].to(device)  # Shape: [1, 1, 28, 28]

model.eval()
with torch.no_grad():
    logits = model(X)              # Shape: [1, 10]
    y_pred = logits.argmax(dim=1)   # Softmax isn't needed for argmax

print(f"Actual class: {labels[0].argmax().item()}")
print(f"Predicted class: {y_pred.item()}")

# ----------------- Train the model ---------------

def train_loop(dataloader, model, loss_fn, optimizer):
    batch_size= len(dataloader.dataset)
    model.train() # Train mode
    for batch, (X,y) in enumerate(dataloader):
        
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
           loss, current = loss.item(), batch * batch_size + len(X)
           print(f"loss: {loss:>7f}  [{current:>5d}/{batch_size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.argmax().item()).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
