import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.transforms import v2

device= torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
    transforms.Lambda(lambda x: x.to(device))
])

target_transform = transforms.Lambda(
    lambda y : F.one_hot(torch.tensor(y), num_classes=10).float().to(device)
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
    transform=transform,
    target_transform=target_transform
)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True,  pin_memory=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

training_dataset_size = len(training_data)
testing_dataset_size = len(test_data)
print(f"Train Dataset size: {training_dataset_size}")
print(f"Test Dataset size: {testing_dataset_size}")
