"""
Main federated learning module integrating MPC and ZKP components.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from .zkp import ComplianceCircuit, Proof, ProofWitness, ComplianceConstraint
import hashlib
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
import time
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('zkml.federated')

@dataclass
class TrainingMetadata:
    """Metadata for a training round."""
    round_number: int
    num_samples: int
    loss: float
    metrics: Dict[str, float]
    timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            'round_number': self.round_number,
            'num_samples': self.num_samples,
            'loss': self.loss,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }

@dataclass
class GradientUpdate:
    """Secure gradient update with compliance proof."""
    model_update: Dict[str, torch.Tensor]  # Parameter updates
    proof_data: Dict[str, Any]  # Proof data for verification
    metadata: TrainingMetadata

class SecureFederatedLearner:
    """Main class for secure federated learning with compliance proofs."""
    
    def __init__(self,
                 model: nn.Module,
                 num_parties: int,
                 compliance_constraints: List[ComplianceConstraint],
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 clip_norm: float = 1.0,
                 secure_aggregation: bool = True,
                 checkpoint_dir: Optional[str] = None):
        """Initialize secure federated learner.
        
        Args:
            model: PyTorch model to train
            num_parties: Number of participating parties
            compliance_constraints: List of regulatory constraints
            dp_epsilon: Differential privacy budget
            dp_delta: DP failure probability
            learning_rate: Learning rate for local optimizers
            weight_decay: Weight decay for regularization
            clip_norm: Gradient clipping norm
            secure_aggregation: Whether to use secure aggregation
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.num_parties = num_parties
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.secure_aggregation = secure_aggregation
        
        # Initialize ZKP components
        self.compliance_circuit = ComplianceCircuit(compliance_constraints)
        
        # Setup checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Initialize round counter
        self.current_round = 0
        
        # Initialize metrics tracking
        self.metrics_history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'f1_score': [],
            'dp_noise_scale': []
        }
        
        logger.info(f"Initialized SecureFederatedLearner with {num_parties} parties")
        logger.info(f"DP parameters: epsilon={dp_epsilon}, delta={dp_delta}")
        logger.info(f"Secure aggregation: {secure_aggregation}")
    
    def compute_local_update(self, 
                           data: torch.Tensor, 
                           labels: torch.Tensor, 
                           batch_size: int = 32, 
                           num_epochs: int = 3,
                           class_weights: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, Any]]:
        """Compute local update with differential privacy and compliance proof.
        
        Args:
            data: Training data tensor
            labels: Ground truth labels
            batch_size: Mini-batch size
            num_epochs: Number of local epochs
            class_weights: Optional weights for imbalanced classes
            
        Returns:
            Tuple of (model update, loss value, proof data)
        """
        logger.info(f"Computing local update on {len(data)} samples")
        start_time = time.time()
        
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Create data loader for mini-batches
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=False,
            num_workers=0
        )
        
        # Calculate class weights if not provided
        if class_weights is None and len(torch.unique(labels)) <= 10:  # Only for classification tasks
            class_counts = torch.bincount(labels)
            total_samples = len(labels)
            class_weights = total_samples / (class_counts.float() * len(class_counts))
            logger.info(f"Calculated class weights: {class_weights}")
        
        total_loss = 0.0
        num_batches = 0
        
        # Train for multiple epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            # Use tqdm for progress tracking
            for batch_data, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_data)
                
                # Calculate sample weights based on class
                if class_weights is not None:
                    sample_weights = torch.tensor([class_weights[label] for label in batch_labels])
                else:
                    sample_weights = torch.ones(len(batch_labels))
                
                # Label smoothing for better generalization
                if outputs.shape[1] > 1:  # Multi-class classification
                    smooth_labels = torch.zeros_like(outputs)
                    smooth_labels.scatter_(1, batch_labels.unsqueeze(1), 0.9)
                    smooth_labels += 0.1 / outputs.size(1)
                    
                    # Custom loss with label smoothing and class weights
                    log_probs = F.log_softmax(outputs, dim=1)
                    weighted_loss = -(smooth_labels * log_probs).sum(dim=1) * sample_weights
                    loss = weighted_loss.mean()
                else:  # Binary classification or regression
                    if outputs.shape[1] == 1:  # Binary classification
                        loss = F.binary_cross_entropy_with_logits(
                            outputs.squeeze(), 
                            batch_labels.float(),
                            weight=sample_weights
                        )
                    else:  # Regression
                        loss = F.mse_loss(outputs.squeeze(), batch_labels.float())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            # Average loss for this epoch
            avg_epoch_loss = epoch_loss / epoch_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
            num_batches += 1
        
        # Compute average loss across all epochs
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get model update (difference between current and original parameters)
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Extract model update
        update = {}
        grad_scale = 0.01  # Scale factor for gradients
        
        # Calculate sensitivity for DP
        sensitivity = self._compute_gradient_sensitivity() * grad_scale
        noise_std = (sensitivity * np.sqrt(2 * np.log(1.25/self.dp_delta))) / self.dp_epsilon
        
        # Add noise to each parameter for differential privacy
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Calculate update
                update[name] = (param.data - original_state[name]) * grad_scale
                
                # Add calibrated noise for differential privacy
                if noise_std > 0:
                    noise = torch.randn_like(update[name]) * noise_std
                    update[name] += noise
        
        # Create compliance proof data
        proof_data = {
            'num_samples': len(data),
            'noise_std': noise_std,
            'sensitivity': sensitivity,
            'grad_scale': grad_scale,
            'data_hash': self._hash_data(data).hex()
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Local update computed in {elapsed_time:.2f} seconds")
        logger.info(f"Average loss: {avg_loss:.4f}, DP noise scale: {noise_std:.6f}")
        
        return update, avg_loss, proof_data
    
    def aggregate_updates(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Securely aggregate updates from all parties.
        
        Args:
            updates: List of model updates
            
        Returns:
            Aggregated model update
        """
        logger.info(f"Aggregating updates from {len(updates)} parties")
        
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        # Verify that all updates have the same structure
        first_update = updates[0]
        for update in updates[1:]:
            if set(update.keys()) != set(first_update.keys()):
                raise ValueError("Updates have inconsistent parameter structures")
        
        # Aggregate parameter updates
        aggregated_update = {}
        
        # Simple averaging (in practice, this would use secure MPC)
        for param_name in first_update.keys():
            param_updates = []
            for update in updates:
                param_updates.append(update[param_name])
            
            # Average the updates
            aggregated_update[param_name] = torch.stack(param_updates).mean(dim=0)
        
        logger.info("Updates aggregated successfully")
        return aggregated_update
    
    def apply_update(self, update: Dict[str, torch.Tensor]) -> None:
        """Apply aggregated update to model parameters.
        
        Args:
            update: Aggregated parameter updates
        """
        logger.info("Applying aggregated update to model")
        
        # Apply updates to corresponding model parameters
        for name, param in self.model.named_parameters():
            if name in update and param.requires_grad:
                param.data.add_(update[name])
        
        # Increment round counter
        self.current_round += 1
        
        # Save checkpoint if directory is specified
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_round_{self.current_round}.pt")
            self.save_checkpoint(checkpoint_path)
            
        logger.info(f"Update applied, completed round {self.current_round}")
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'round': self.current_round,
            'metrics_history': self.metrics_history,
            'dp_epsilon': self.dp_epsilon,
            'dp_delta': self.dp_delta
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_round = checkpoint['round']
        self.metrics_history = checkpoint['metrics_history']
        self.dp_epsilon = checkpoint['dp_epsilon']
        self.dp_delta = checkpoint['dp_delta']
        
        logger.info(f"Checkpoint loaded from {path} (round {self.current_round})")
    
    def evaluate(self, 
                data: torch.Tensor, 
                labels: torch.Tensor, 
                batch_size: int = 256) -> Dict[str, float]:
        """Evaluate model on validation/test data.
        
        Args:
            data: Evaluation data
            labels: Ground truth labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(data)} samples")
        self.model.eval()
        
        # Create data loader
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_outputs = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                outputs = self.model(batch_data)
                
                # Calculate loss
                if outputs.shape[1] > 1:  # Multi-class
                    loss = F.cross_entropy(outputs, batch_labels)
                    _, predicted = torch.max(outputs, 1)
                else:  # Binary or regression
                    if outputs.shape[1] == 1:  # Binary
                        loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), batch_labels.float())
                        predicted = (outputs.squeeze() > 0).float()
                    else:  # Regression
                        loss = F.mse_loss(outputs.squeeze(), batch_labels.float())
                        predicted = outputs.squeeze()
                
                total_loss += loss.item() * len(batch_data)
                all_outputs.append(predicted)
                all_labels.append(batch_labels)
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        metrics = {}
        metrics['loss'] = total_loss / len(data)
        
        # Classification metrics
        if outputs.shape[1] <= 10:  # Classification task
            metrics['accuracy'] = (all_outputs == all_labels).float().mean().item()
            
            # For binary classification, calculate more metrics
            if outputs.shape[1] == 1 or outputs.shape[1] == 2:
                true_positives = ((all_outputs == 1) & (all_labels == 1)).sum().item()
                false_positives = ((all_outputs == 1) & (all_labels == 0)).sum().item()
                false_negatives = ((all_outputs == 0) & (all_labels == 1)).sum().item()
                true_negatives = ((all_outputs == 0) & (all_labels == 0)).sum().item()
                
                # Precision, recall, F1
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1
                metrics['true_positives'] = true_positives
                metrics['false_positives'] = false_positives
                metrics['false_negatives'] = false_negatives
                metrics['true_negatives'] = true_negatives
        
        # Update metrics history
        self.metrics_history['round'].append(self.current_round)
        self.metrics_history['loss'].append(metrics.get('loss', 0))
        self.metrics_history['accuracy'].append(metrics.get('accuracy', 0))
        self.metrics_history['f1_score'].append(metrics.get('f1_score', 0))
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def _compute_gradient_sensitivity(self) -> float:
        """Compute L2 sensitivity of gradient computation."""
        # For clipped gradients, sensitivity is the clipping norm
        return self.clip_norm
    
    def _hash_data(self, data: torch.Tensor) -> bytes:
        """Hash training data for ZKP."""
        data_bytes = data.numpy().tobytes()
        return hashlib.sha256(data_bytes).digest()
    
    def verify_compliance(self, proof_data: Dict[str, Any]) -> bool:
        """Verify compliance with regulatory constraints.
        
        Args:
            proof_data: Proof data from client
            
        Returns:
            True if compliant, False otherwise
        """
        # Create a witness from proof data
        witness = ProofWitness(
            data_hash=bytes.fromhex(proof_data['data_hash']),
            field_values={'num_samples': proof_data['num_samples']},
            noise_params={'noise_std': proof_data['noise_std']}
        )
        
        # Verify using compliance circuit
        try:
            return self.compliance_circuit.verify_proof(
                Proof(a=(0,0,0), b=(0,0,0), c=(0,0,0), witness=witness),
                {
                    'num_samples': proof_data['num_samples'],
                    'noise_std': proof_data['noise_std']
                }
            )
        except Exception as e:
            logger.error(f"Error verifying compliance: {str(e)}")
            return False 