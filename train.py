import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from autoencoder import Autoencoder

def train_autoencoders(train_dataloaders):    
    # Initialize one autoencoder per digit
    autoencoders = {digit: Autoencoder(784, 512, 256, 128) for digit in range(10)}

    # Training each autoencoder
    loss_function = nn.MSELoss()
    epochs = 20

    for digit, model in autoencoders.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print(f"Training autoencoder for digit {digit}")
        for epoch in range(epochs):
            train_loss = 0
            for data, _ in train_dataloaders[digit]:
                #data = data.view(-1, 784).cuda()
                data = data.view(-1, 784)
                optimizer.zero_grad()
                recon = model(data)
                loss = loss_function(recon, data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
    return autoencoders