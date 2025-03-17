"""
ZKML: Secure Multiparty Machine Learning with Zero-Knowledge Compliance Proofs
"""

from .models import MedicalDiagnosisModel
from .federated import SecureFederatedLearner, ComplianceConstraint
from .zkp import ProofWitness, ComplianceCircuit
from .mpc import SecretSharing

__version__ = "0.1.0"

__all__ = [
    'MedicalDiagnosisModel',
    'SecureFederatedLearner',
    'ComplianceConstraint',
    'ProofWitness',
    'ComplianceCircuit',
    'SecretSharing'
] 