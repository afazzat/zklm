# ZKML: Secure Federated Learning with Zero-Knowledge Compliance

ZKML is a Python library that enables privacy-preserving collaborative machine learning by combining secure multi-party computation (MPC) and zero-knowledge proofs (ZKP). The framework allows multiple parties to jointly train neural networks while:

1. Keeping raw data private through MPC-based gradient aggregation
2. Ensuring regulatory compliance through succinct zero-knowledge proofs
3. Providing differential privacy guarantees for model updates

## Features

- **Secure Gradient Aggregation**: Uses an optimized SPDZ protocol for secure aggregation of model updates
- **Compliance Verification**: Generates and verifies Groth16 proofs for regulatory constraints
- **Differential Privacy**: Implements the Gaussian mechanism with automatic privacy parameter calibration
- **PyTorch Integration**: Seamlessly works with PyTorch models and datasets
- **Modern Cryptography**: Built on the BN254 curve with state-of-the-art cryptographic primitives

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/zkml.git
cd zkml

# Install dependencies
pip install -e .
```

## Quick Start

Here's a simple example of how to use ZKML for secure federated learning on MNIST:

```python
from zkml.federated import SecureFederatedLearner
from zkml.zkp import ComplianceConstraint
import torch.nn as nn

# Define your model
model = nn.Sequential(...)

# Define compliance constraints
constraints = [
    ComplianceConstraint(
        name="sample_size",
        min_value=100,
        max_value=float('inf'),
        required_fields=["num_samples"]
    )
]

# Initialize secure federated learner
learner = SecureFederatedLearner(
    model=model,
    num_parties=4,
    compliance_constraints=constraints,
    dp_epsilon=1.0,
    dp_delta=1e-5
)

# Train securely
for round in range(num_rounds):
    # Each party computes local update
    update, loss = learner.compute_local_update(data, labels)
    
    # Aggregate updates securely
    aggregated = learner.aggregate_updates([update1, update2, ...])
    
    # Apply update
    learner.apply_update(aggregated)
```

For a complete example, see `examples/mnist_federated.py`.

## Architecture

The framework consists of three main components:

1. **MPC Module** (`zkml/mpc.py`):
   - Implements secure gradient aggregation using SPDZ
   - Handles secret sharing and MAC-based verification
   - Optimized for efficient linear algebra operations

2. **ZKP Module** (`zkml/zkp.py`):
   - Implements Groth16 proofs over BN254 curve
   - Supports custom regulatory constraints
   - Includes specialized circuits for differential privacy verification

3. **Federated Learning** (`zkml/federated.py`):
   - Coordinates secure training process
   - Integrates MPC and ZKP components
   - Manages model updates and verification

## Security Guarantees

- **Privacy**: Secure against t-out-of-n malicious parties (t < n/2)
- **Integrity**: Zero-knowledge proofs ensure regulatory compliance
- **Differential Privacy**: ε-δ privacy guarantees for model updates

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{zkml2024,
  title={Secure Multiparty Machine Learning with Zero-Knowledge Compliance Proofs},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 