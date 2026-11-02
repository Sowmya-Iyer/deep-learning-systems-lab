import torch
from torch import nn
from data_utils.datasets import train_dataloader, test_dataloader, device


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # [batch, 16, 14, 14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # [batch, 32, 7, 7]

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


model = SmallCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_loop():
    model.train()
    total_loss = 0.0

    for X, y in train_dataloader:
        # Your dataset transforms already move X and y to device.
        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(train_dataloader.dataset)


def test_loop():
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)

            # Convert your one-hot labels to class indices for accuracy.
            correct += (
                logits.argmax(dim=1) == y.argmax(dim=1)
            ).sum().item()

    size = len(test_dataloader.dataset)
    return total_loss / size, correct / size


for epoch in range(1, 6):
    train_loss = train_loop()
    test_loss, accuracy = test_loop()

    print(
        f"Epoch {epoch} | Train loss: {train_loss:.4f} | "
        f"Test loss: {test_loss:.4f} | Accuracy: {accuracy:.2%}"
    )

torch.save(model.state_dict(), "fashion_mnist_cnn.pth")