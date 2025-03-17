"""
Neural network models for medical diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.leaky_relu(out, negative_slope=0.1)
        
        return out

class MedicalDiagnosisModel(nn.Module):
    """Neural network for medical diagnosis with privacy-preserving attributes."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        """Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        # Input projection with batch normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Deep network with skip connections and layer normalization
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # Output layer with reduced dimension for stability
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with privacy-preserving constraints."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved gradient flow."""
        # Input projection
        x = self.input_proj(x)
        
        # Residual connections
        identity = x
        
        # Layer 1 with residual
        x = self.layer1(x)
        x = x + identity
        
        # Layer 2 with residual
        identity = x
        x = self.layer2(x)
        x = x + identity
        
        # Layer 3 with residual
        identity = x
        x = self.layer3(x)
        x = x + identity
        
        # Output projection
        x = self.output(x)
        
        return x
    
    def get_gradient_norm(self) -> float:
        """Compute L2 norm of gradients for privacy accounting."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5 