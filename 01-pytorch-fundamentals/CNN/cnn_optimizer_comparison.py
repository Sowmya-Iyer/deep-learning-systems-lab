import torch
from torch import nn
from data_utils.datasets import train_dataloader, test_dataloader, device

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers= nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),# [batch, 16, 14, 14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # [batch, 32, 7, 7]

            nn.Flatten(),
            nn.Linear(32 * 7* 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(x)


loss_fn = nn.CrossEntropyLoss()


def train_loop(optimizer):
    model.train()
    total_loss = 0.0
    for X,y in train_dataloader:
          optimizer.zero_grad()
          pred= model(X)
          loss = loss_fn(pred, y)

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


learning_rates = [0.01, 0.001, 0.0001]
seeds = [42, 123, 2026]
epochs = 5
results = []

for seed in seeds:
    for lr in learning_rates:
        for optimizer_name in ["SGD", "AdamW"]:
            # Reset randomness before creating each model.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Fresh model: matching initial weights for each paired run.
            model = SmallCNN().to(device)

            if optimizer_name == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=1e-4,
                )

            # Reset the shuffled training order for each run.
            train_dataloader.generator = torch.Generator().manual_seed(seed)
            train_dataloader.sampler.generator = (
                torch.Generator().manual_seed(seed)
            )

            print(
                f"\nOptimizer: {optimizer_name} | "
                f"Learning rate: {lr} | Seed: {seed}"
            )

            for epoch in range(1, epochs + 1):
                train_loss = train_loop(optimizer)
                test_loss, accuracy = test_loop()

                print(
                    f"Epoch {epoch} | Train loss: {train_loss:.4f} | "
                    f"Test loss: {test_loss:.4f} | "
                    f"Accuracy: {accuracy:.2%}"
                )

                results.append({
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "accuracy": accuracy,
                })

            torch.save(
                model.state_dict(),
                f"cnn_{optimizer_name}_lr{lr}_seed{seed}.pth",
            )

# Save measurements for comparison.
import csv

with open("cnn_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)