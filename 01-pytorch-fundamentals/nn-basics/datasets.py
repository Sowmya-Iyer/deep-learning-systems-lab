import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.transforms import v2

transform = transforms.Compose([
    transforms.ToTensor()
])
target_transform = transforms.Lambda(
    lambda y : F.one_hot(torch.tensor(y), num_classes=10).float()
)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=target_transform)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
))


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

training_dataset_size = len(training_data)
testing_dataset_size = len(test_data)
print(f"Train Dataset size: {training_dataset_size}")
print(f"Test Dataset size: {testing_dataset_size}")
