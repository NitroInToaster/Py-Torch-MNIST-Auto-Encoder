import numpy as np
import torch
from torch.utils.data import DataLoader

def evaluate_models(autoencoders, full_test_dataset):
    batch_size = 128
    # Evaluation
    print("Evaluating models...")
    total, correct = 0, 0
    for data, labels in DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False):
        #data = data.view(-1, 784).cuda()
        data = data.view(-1, 784)
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

    return accuracy