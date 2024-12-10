import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def prepare_dataloaders():    
    # Load MNIST dataset
    transform = transforms.ToTensor()
    full_train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    full_test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=True)

    # Create datasets and dataloaders for each digit
    train_dataloaders = {}
    test_dataloaders = {}
    batch_size = 128

    for digit in range(10):
        train_indices = [i for i, target in enumerate(full_train_dataset.targets) if target == digit]
        test_indices = [i for i, target in enumerate(full_test_dataset.targets) if target == digit]
        train_dataloaders[digit] = DataLoader(Subset(full_train_dataset, train_indices), batch_size=batch_size, shuffle=True)
        test_dataloaders[digit] = DataLoader(Subset(full_test_dataset, test_indices), batch_size=batch_size, shuffle=False)
    
    return train_dataloaders, test_dataloaders, full_test_dataset