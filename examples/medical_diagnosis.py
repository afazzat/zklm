"""
Example of secure federated learning for medical diagnosis.

This example demonstrates a practical implementation of federated learning
for a medical diagnosis task with differential privacy and zero-knowledge proofs.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
from tqdm import tqdm
import traceback
import sys
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('medical_diagnosis.log')
    ]
)
logger = logging.getLogger('medical_diagnosis')

# Create results directory
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(results_dir, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
logger.info(f"Results will be saved to: {run_dir}")

try:
    logger.info("Loading zkml modules...")
    from zkml.models import MedicalDiagnosisModel
    from zkml.federated import SecureFederatedLearner, ComplianceConstraint
    
    # Set random seeds for reproducibility
    def set_seeds(seed: int = 42) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seeds(42)
    
    def generate_correlated_features(num_samples: int, correlation_matrix: np.ndarray) -> np.ndarray:
        """Generate correlated features using Cholesky decomposition."""
        num_features = correlation_matrix.shape[0]
        L = np.linalg.cholesky(correlation_matrix)
        uncorrelated = np.random.standard_normal((num_samples, num_features))
        return uncorrelated @ L.T
    
    def load_synthetic_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic medical data with realistic patterns and correlations."""
        logger.info(f"Generating {num_samples} synthetic medical data samples...")
        
        # Define correlation matrix for key medical indicators
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3, 0.4, 0.2],  # Age
            [0.5, 1.0, 0.4, 0.3, 0.3],  # Blood Pressure
            [0.3, 0.4, 1.0, 0.6, 0.4],  # Glucose
            [0.4, 0.3, 0.6, 1.0, 0.5],  # BMI
            [0.2, 0.3, 0.4, 0.5, 1.0]   # Cholesterol
        ])
        
        # Generate correlated base features
        base_features = generate_correlated_features(num_samples, correlation_matrix)
        
        # Transform to realistic ranges
        age = stats.norm.ppf(stats.norm.cdf(base_features[:, 0])) * 15 + 50  # Age: mean 50, std 15
        blood_pressure = stats.norm.ppf(stats.norm.cdf(base_features[:, 1])) * 20 + 120  # BP: mean 120, std 20
        glucose = stats.norm.ppf(stats.norm.cdf(base_features[:, 2])) * 25 + 100  # Glucose: mean 100, std 25
        bmi = stats.norm.ppf(stats.norm.cdf(base_features[:, 3])) * 5 + 25  # BMI: mean 25, std 5
        cholesterol = stats.norm.ppf(stats.norm.cdf(base_features[:, 4])) * 40 + 200  # Cholesterol: mean 200, std 40
        
        # Generate additional features with medical relevance
        heart_rate = np.random.normal(75, 12, num_samples)  # Heart rate: mean 75, std 12
        crp = np.random.gamma(2, 3, num_samples)  # C-reactive protein (inflammation)
        wbc = np.random.normal(7.5, 2, num_samples)  # White blood cell count
        temp = np.random.normal(37, 0.7, num_samples)  # Body temperature
        hdl = np.random.normal(50, 15, num_samples)  # HDL Cholesterol
        ldl = cholesterol - hdl  # LDL Cholesterol (dependent on total cholesterol)
        triglycerides = np.random.gamma(shape=2, scale=50, size=num_samples)  # Triglycerides
        
        # Create feature matrix
        X = np.column_stack([
            age, blood_pressure, glucose, bmi, cholesterol,
            heart_rate, crp, wbc, temp, hdl, ldl, triglycerides
        ])
        
        # Add derived features and interactions
        X = np.column_stack([
            X,
            (age * bmi) / 1000,  # Age-BMI interaction
            (blood_pressure * glucose) / 10000,  # BP-Glucose interaction
            np.sin(age/50),  # Non-linear age effect
            np.exp(-bmi/50),  # Non-linear BMI effect
            (cholesterol * bmi) / 10000,  # Cholesterol-BMI interaction
            (blood_pressure * heart_rate) / 10000,  # BP-Heart Rate interaction
            (glucose * bmi) / 1000,  # Glucose-BMI interaction
            np.maximum(0, blood_pressure - 120),  # Hypertension indicator
            np.maximum(0, glucose - 100),  # Hyperglycemia indicator
            np.maximum(0, bmi - 25),  # Overweight indicator
            np.maximum(0, crp - 3),  # Inflammation indicator
            np.maximum(0, wbc - 10),  # High WBC indicator
            np.maximum(0, temp - 37.5),  # Fever indicator
            ldl / hdl,  # Cholesterol ratio
            triglycerides / hdl  # Triglyceride-HDL ratio
        ])
        
        # Calculate risk score using medical domain knowledge
        risk_score = (
            0.3 * stats.norm.cdf((age - 50) / 15) +  # Age risk
            0.15 * stats.norm.cdf((blood_pressure - 120) / 20) +  # BP risk
            0.15 * stats.norm.cdf((glucose - 100) / 25) +  # Glucose risk
            0.1 * stats.norm.cdf((bmi - 25) / 5) +  # BMI risk
            0.1 * stats.norm.cdf((cholesterol - 200) / 40) +  # Cholesterol risk
            0.05 * stats.norm.cdf((crp - 2) / 3) +  # Inflammation risk
            0.05 * stats.norm.cdf((wbc - 7.5) / 2) +  # WBC risk
            0.05 * stats.norm.cdf((temp - 37) / 0.7) +  # Temperature risk
            0.05 * stats.norm.cdf((ldl/hdl - 3) / 1)  # Cholesterol ratio risk
        )
        
        # Add some randomness for genetic/environmental factors
        risk_score += np.random.normal(0, 0.1, num_samples)
        
        # Generate binary labels with realistic prevalence
        disease_prevalence = 0.2  # 20% positive cases
        threshold = np.percentile(risk_score, 100 * (1 - disease_prevalence))
        y = (risk_score > threshold).astype(int)
        
        # Log data statistics
        logger.info(f"Generated features shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        logger.info(f"Feature ranges:\n{pd.DataFrame(X).describe()}")
        
        # Save feature importance plot
        feature_importance = np.abs(np.corrcoef(X.T, risk_score)[-1, :-1])
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title("Feature Importance (Correlation with Risk Score)")
        plt.xlabel("Feature Index")
        plt.ylabel("Absolute Correlation")
        plt.savefig(os.path.join(run_dir, "feature_importance.png"))
        plt.close()
        
        return X, y
    
    def split_data(X: np.ndarray, y: np.ndarray, num_parties: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data among parties while preserving class distribution."""
        logger.info(f"Splitting data among {num_parties} parties...")
        
        # First split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features using robust scaler (less sensitive to outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for future use
        np.save(os.path.join(run_dir, "scaler_center_.npy"), scaler.center_)
        np.save(os.path.join(run_dir, "scaler_scale_.npy"), scaler.scale_)
        
        # Split training data among parties
        party_data = []
        
        # Calculate sizes for each party (equal split)
        party_size = len(X_train) // num_parties
        remaining = len(X_train) % num_parties
        
        start_idx = 0
        for i in range(num_parties):
            # Add one extra sample if there are remaining samples
            current_size = party_size + (1 if i < remaining else 0)
            end_idx = start_idx + current_size
            
            # Get stratified subset for this party
            party_X = X_train_scaled[start_idx:end_idx]
            party_y = y_train[start_idx:end_idx]
            
            party_data.append((party_X, party_y))
            start_idx = end_idx
            
            # Log party data statistics
            logger.info(f"Party {i} data shape: {party_X.shape}")
            logger.info(f"Party {i} class distribution: {np.bincount(party_y)}")
        
        # Save test data
        np.save(os.path.join(run_dir, "X_test.npy"), X_test_scaled)
        np.save(os.path.join(run_dir, "y_test.npy"), y_test)
        
        return party_data, (X_test_scaled, y_test)
    
    def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Get predictions and probabilities
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Calculate metrics
            accuracy = (preds == y_tensor).float().mean().item()
            
            # ROC curve and AUC
            fpr, tpr, _ = roc_curve(y, probs[:, 1].numpy())
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve and AUC
            precision, recall, _ = precision_recall_curve(y, probs[:, 1].numpy())
            pr_auc = auc(recall, precision)
            
            # Calculate F1 score
            tp = ((preds == 1) & (y_tensor == 1)).sum().item()
            fp = ((preds == 1) & (y_tensor == 0)).sum().item()
            fn = ((preds == 0) & (y_tensor == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Save ROC curve plot
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(run_dir, "roc_curve.png"))
            plt.close()
            
            # Save Precision-Recall curve plot
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(run_dir, "pr_curve.png"))
            plt.close()
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
    
    def main():
        try:
            # Generate synthetic data
            num_samples = 5000
            X, y = load_synthetic_data(num_samples)
            
            # Split data among parties
            num_parties = 3
            party_data, (X_test, y_test) = split_data(X, y, num_parties)
            
            # Initialize model with improved architecture
            input_dim = X.shape[1]
            hidden_dim = 1024
            model = MedicalDiagnosisModel(input_dim=input_dim, hidden_dim=hidden_dim)
            
            # Define compliance constraints
            constraints = [
                ComplianceConstraint(
                    name="min_samples",
                    condition="num_samples >= 100",
                    description="Minimum number of samples per party",
                    verification_type="range",
                    parameters={"field": "num_samples", "min": 100}
                ),
                ComplianceConstraint(
                    name="dp_noise",
                    condition="noise_std >= 0.1",
                    description="Minimum noise for differential privacy",
                    verification_type="threshold",
                    parameters={"field": "noise_std", "threshold": 0.1}
                )
            ]
            
            # Initialize federated learner with improved parameters
            learner = SecureFederatedLearner(
                model=model,
                num_parties=num_parties,
                learning_rate=0.001,
                dp_epsilon=1.0,
                dp_delta=1e-5,
                compliance_constraints=constraints,
                clip_norm=1.0,
                weight_decay=1e-5,
                secure_aggregation=True,
                checkpoint_dir=run_dir
            )
            
            # Training history
            history = {
                'train_loss': [],
                'train_metrics': [],
                'test_metrics': [],
                'best_test_metrics': None,
                'best_round': 0
            }
            
            # Early stopping parameters
            patience = 5
            best_f1 = 0
            rounds_without_improvement = 0
            
            # Train federated model
            logger.info("Starting federated training...")
            for round_idx in tqdm(range(50), desc="Training rounds"):  # 50 rounds
                # Collect updates from all parties
                round_losses = []
                round_updates = []
                for party_idx, (X_party, y_party) in enumerate(party_data):
                    update, loss, proof = learner.compute_local_update(
                        data=torch.FloatTensor(X_party),
                        labels=torch.LongTensor(y_party),
                        batch_size=128,
                        num_epochs=3,
                        class_weights=None
                    )
                    round_losses.append(loss)
                    round_updates.append(update)
                
                # Average loss across parties
                avg_loss = np.mean(round_losses)
                history['train_loss'].append(avg_loss)
                
                # Aggregate model updates
                learner.aggregate_updates(round_updates)
                
                # Evaluate on test set
                test_metrics = evaluate_model(model, X_test, y_test)
                history['test_metrics'].append(test_metrics)
                
                # Update best metrics
                if test_metrics['f1'] > best_f1:
                    best_f1 = test_metrics['f1']
                    history['best_test_metrics'] = test_metrics
                    history['best_round'] = round_idx
                    rounds_without_improvement = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
                else:
                    rounds_without_improvement += 1
                
                # Log progress
                logger.info(f"Round {round_idx + 1}/{50}")
                logger.info(f"Average Loss: {avg_loss:.4f}")
                logger.info(f"Test Metrics: {test_metrics}")
                
                # Early stopping
                if rounds_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {round_idx + 1} rounds")
                    break
            
            # Plot training history
            plt.figure(figsize=(12, 6))
            plt.plot(history['train_loss'], label='Training Loss')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.savefig(os.path.join(run_dir, "training_loss.png"))
            plt.close()
            
            # Plot metrics history
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
            plt.figure(figsize=(15, 10))
            for i, metric in enumerate(metrics):
                plt.subplot(2, 3, i+1)
                values = [m[metric] for m in history['test_metrics']]
                plt.plot(values, label=metric)
                plt.xlabel('Round')
                plt.ylabel(metric.capitalize())
                plt.title(f'{metric.upper()} Over Time')
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "metrics_history.png"))
            plt.close()
            
            # Log final results
            logger.info("\nTraining completed!")
            logger.info(f"Best performance at round {history['best_round'] + 1}")
            logger.info("Best test metrics:")
            for metric, value in history['best_test_metrics'].items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save training history
            np.save(os.path.join(run_dir, "training_history.npy"), history)
            
        except Exception as e:
            logger.error("Error during training:")
            logger.error(traceback.format_exc())
            raise e

    if __name__ == "__main__":
        main()
        
except Exception as e:
    # Catch any high-level exception for better error reporting
    logger.error(f"ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 