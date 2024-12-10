import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

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

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3):
        super(Autoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        # Decoder
        self.fc4 = nn.Linear(h_dim3, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def decoder(self, x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize one autoencoder per digit
autoencoders = {digit: Autoencoder(784, 512, 256, 128).cuda() for digit in range(10)}
#autoencoders = {digit: Autoencoder(784, 512, 256, 128) for digit in range(10)}

# Training each autoencoder
loss_function = nn.MSELoss()
epochs = 20

for digit, model in autoencoders.items():
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training autoencoder for digit {digit}")
    for epoch in range(epochs):
        train_loss = 0
        for data, _ in train_dataloaders[digit]:
            data = data.view(-1, 784).cuda()
            #data = data.view(-1, 784)
            optimizer.zero_grad()
            recon = model(data)
            loss = loss_function(recon, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluation
print("Evaluating models...")
total, correct = 0, 0
for data, labels in DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False):
    data = data.view(-1, 784).cuda()
    #data = data.view(-1, 784)
    reconstruction_errors = []

    # Calculate reconstruction errors for each model
    for digit, model in autoencoders.items():
        recon = model(data)
        errors = torch.mean((data - recon) ** 2, dim=1)  # Reconstruction error
        reconstruction_errors.append(errors.cpu().detach().numpy())

    # Identify the model with the smallest error
    reconstruction_errors = np.stack(reconstruction_errors, axis=1)
    predictions = np.argmin(reconstruction_errors, axis=1)

    # Calculate accuracy
    correct += (predictions == labels.numpy()).sum()
    total += labels.size(0)

accuracy = 100 * correct / total
print(f"Overall recognition accuracy: {accuracy:.2f}%")
