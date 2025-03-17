"""
Example of secure federated learning on MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np

from zkml.federated import SecureFederatedLearner
from zkml.zkp import ComplianceConstraint

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_and_split_mnist(num_parties: int):
    """Load MNIST and split into num_parties parts."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True,
                           transform=transform)
    
    # Split dataset
    n = len(dataset)
    party_size = n // num_parties
    party_datasets = random_split(
        dataset, 
        [party_size] * (num_parties - 1) + [n - party_size * (num_parties - 1)]
    )
    
    return party_datasets

def main():
    # Parameters
    num_parties = 4
    num_rounds = 10
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define compliance constraints
    constraints = [
        ComplianceConstraint(
            name="sample_size",
            min_value=100,  # Minimum samples per party
            max_value=float('inf'),
            required_fields=["num_samples"]
        )
    ]
    
    # Initialize model
    model = SimpleCNN().to(device)
    
    # Initialize secure federated learner
    learner = SecureFederatedLearner(
        model=model,
        num_parties=num_parties,
        compliance_constraints=constraints,
        dp_epsilon=1.0,
        dp_delta=1e-5
    )
    
    # Load and split data
    party_datasets = load_and_split_mnist(num_parties)
    
    # Training loop
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        
        # Collect updates from all parties
        updates = []
        losses = []
        
        for party_idx, dataset in enumerate(party_datasets):
            print(f"Party {party_idx + 1} computing update...")
            
            # Get party's data
            data = torch.stack([x for x, _ in dataset])
            labels = torch.tensor([y for _, y in dataset])
            
            # Compute local update
            update, loss = learner.compute_local_update(
                data.to(device),
                labels.to(device),
                batch_size
            )
            updates.append(update)
            losses.append(loss)
            
            print(f"Party {party_idx + 1} loss: {loss:.4f}")
        
        # Aggregate updates
        print("Aggregating updates...")
        aggregated = learner.aggregate_updates(updates)
        
        # Apply update
        learner.apply_update(aggregated)
        
        # Print round summary
        avg_loss = np.mean(losses)
        print(f"Round {round_num + 1} complete. Average loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main() 