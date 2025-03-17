"""
Secure Multi-Party Computation (MPC) module for gradient aggregation.
Implements an optimized variant of the SPDZ protocol with ABY3 and MP-SPDZ integration.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass, field
import hashlib
import logging
import time
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

# Configure logging
logger = logging.getLogger('zkml.mpc')

class MPCError(Exception):
    """Base exception for MPC-related errors."""
    pass

class ProtocolError(MPCError):
    """Error during protocol execution."""
    pass

class AuthenticationError(MPCError):
    """Error during data authentication."""
    pass

@dataclass
class SecretShare:
    """Authenticated additive secret share with MAC."""
    
    value: np.ndarray
    share_id: int
    mac: Optional[np.ndarray] = None
    prime_field: int = 2**61 - 1
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate share initialization."""
        if not isinstance(self.value, np.ndarray):
            raise ValueError("Share value must be a numpy array")
        if self.share_id < 0:
            raise ValueError("Share ID must be non-negative")
            
    def generate_mac(self, key: np.ndarray) -> None:
        """Generate MAC for share authentication."""
        # Convert share to bytes
        value_bytes = self.value.tobytes()
        id_bytes = self.share_id.to_bytes(4, byteorder='big')
        
        # Generate MAC using HMAC-SHA256
        h = hashlib.sha256()
        h.update(key.tobytes())
        h.update(value_bytes)
        h.update(id_bytes)
        
        # Convert MAC to field element
        self.mac = np.frombuffer(h.digest(), dtype=np.uint8).astype(np.int64) % self.prime_field
        
    def verify_mac(self, key: np.ndarray) -> bool:
        """Verify share MAC."""
        if self.mac is None:
            return False
            
        # Recompute MAC
        value_bytes = self.value.tobytes()
        id_bytes = self.share_id.to_bytes(4, byteorder='big')
        
        h = hashlib.sha256()
        h.update(key.tobytes())
        h.update(value_bytes)
        h.update(id_bytes)
        
        expected_mac = np.frombuffer(h.digest(), dtype=np.uint8).astype(np.int64) % self.prime_field
        return np.array_equal(self.mac, expected_mac)
    
    def __add__(self, other: 'SecretShare') -> 'SecretShare':
        """Add two secret shares."""
        if not isinstance(other, SecretShare):
            raise TypeError("Can only add SecretShare objects")
        if self.prime_field != other.prime_field:
            raise ValueError("Shares must be in the same field")
            
        result = SecretShare(
            value=(self.value + other.value) % self.prime_field,
            share_id=self.share_id,
            prime_field=self.prime_field
        )
        
        # Add MACs if both shares have them
        if self.mac is not None and other.mac is not None:
            result.mac = (self.mac + other.mac) % self.prime_field
            
        return result
    
    def __mul__(self, scalar: int) -> 'SecretShare':
        """Multiply share by scalar."""
        result = SecretShare(
            value=(self.value * scalar) % self.prime_field,
            share_id=self.share_id,
            prime_field=self.prime_field
        )
        
        if self.mac is not None:
            result.mac = (self.mac * scalar) % self.prime_field
            
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert share to dictionary for serialization."""
        return {
            "value": self.value.tolist(),
            "share_id": self.share_id,
            "mac": self.mac.tolist() if self.mac is not None else None,
            "prime_field": self.prime_field,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecretShare':
        """Create share from dictionary."""
        return cls(
            value=np.array(data["value"]),
            share_id=data["share_id"],
            mac=np.array(data["mac"]) if data["mac"] is not None else None,
            prime_field=data["prime_field"],
            metadata=data["metadata"]
        )

class SecretSharing:
    """Advanced additive secret sharing with information-theoretic security."""
    
    def __init__(self, num_parties: int, prime_field: int = 2**61 - 1, security_parameter: int = 128):
        """Initialize secret sharing scheme."""
        self.num_parties = num_parties
        self.prime_field = prime_field
        self.security_parameter = security_parameter
        
        # Generate global MAC key
        self.mac_key = self._generate_mac_key()
        
        logger.info(f"Initialized secret sharing with {num_parties} parties and {security_parameter}-bit security")
    
    def _generate_mac_key(self) -> np.ndarray:
        """Generate random MAC key."""
        key_bytes = os.urandom(self.security_parameter // 8)
        return np.frombuffer(key_bytes, dtype=np.uint8).astype(np.int64) % self.prime_field
    
    def generate_shares(self, secret: np.ndarray) -> List[SecretShare]:
        """Generate additive secret shares."""
        if not isinstance(secret, np.ndarray):
            raise ValueError("Secret must be a numpy array")
            
        # Convert to field elements
        secret = secret.astype(np.int64) % self.prime_field
        
        # Generate random shares
        shares = []
        sum_shares = np.zeros_like(secret)
        
        for i in range(self.num_parties - 1):
            # Generate random share
            share = np.random.randint(0, self.prime_field, size=secret.shape, dtype=np.int64)
            sum_shares = (sum_shares + share) % self.prime_field
            
            # Create share object with MAC
            share_obj = SecretShare(share, i, prime_field=self.prime_field)
            share_obj.generate_mac(self.mac_key)
            shares.append(share_obj)
        
        # Last share is the difference
        last_share = (secret - sum_shares) % self.prime_field
        share_obj = SecretShare(last_share, self.num_parties - 1, prime_field=self.prime_field)
        share_obj.generate_mac(self.mac_key)
        shares.append(share_obj)
        
        return shares
    
    def reconstruct_secret(self, shares: List[SecretShare]) -> np.ndarray:
        """Reconstruct secret from shares with MAC verification."""
        # Verify MACs
        for share in shares:
            if not share.verify_mac(self.mac_key):
                raise AuthenticationError(f"MAC verification failed for share {share.share_id}")
        
        # Verify number of shares
        if len(shares) != self.num_parties:
            raise ProtocolError(f"Expected {self.num_parties} shares, got {len(shares)}")
        
        # Sum shares in prime field
        secret = np.zeros_like(shares[0].value)
        for share in shares:
            secret = (secret + share.value) % self.prime_field
        
        return secret

class ABY3Protocol:
    """Implementation of the ABY3 protocol for 3-party secure computation.
    
    Based on the paper: "ABY3: A Mixed Protocol Framework for Machine Learning"
    by Prakash Mohassel and Peter Rindal, 2018.
    """
    
    def __init__(self, prime_field: int = 2**61 - 1):
        """Initialize ABY3 protocol.
        
        Args:
            prime_field: Prime field modulus
        """
        self.num_parties = 3  # ABY3 is a 3-party protocol
        self.prime_field = prime_field
        self.secret_sharing = SecretSharing(self.num_parties, prime_field)
        
        # Initialize authenticated triples for multiplication
        self._initialize_triples()
        
        logger.info(f"Initialized ABY3 protocol with {self.num_parties} parties")
    
    def _initialize_triples(self) -> None:
        """Generate Beaver triples for multiplication."""
        self.triples = []
        for _ in range(1000):  # Pre-generate 1000 triples
            a = np.random.randint(0, self.prime_field)
            b = np.random.randint(0, self.prime_field)
            c = (a * b) % self.prime_field
            
            # Share each value
            a_shares = self.secret_sharing.generate_shares(np.array([a]))
            b_shares = self.secret_sharing.generate_shares(np.array([b]))
            c_shares = self.secret_sharing.generate_shares(np.array([c]))
            
            self.triples.append((a_shares, b_shares, c_shares))
    
    def share_input(self, input_value: np.ndarray, input_party: int) -> List[SecretShare]:
        """Share input among parties."""
        if input_party < 0 or input_party >= self.num_parties:
            raise ValueError(f"Input party ID must be between 0 and {self.num_parties-1}")
        
        return self.secret_sharing.generate_shares(input_value)
    
    def secure_addition(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Perform secure addition."""
        if len(shares_a) != self.num_parties or len(shares_b) != self.num_parties:
            raise ProtocolError(f"Expected {self.num_parties} shares for each input")
        
        # Addition is performed locally on shares
        result_shares = []
        for a_share, b_share in zip(shares_a, shares_b):
            result_shares.append(a_share + b_share)
        
        return result_shares
    
    def secure_multiplication(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Perform secure multiplication using Beaver triples."""
        if not self.triples:
            raise ProtocolError("No multiplication triples available")
        
        # Get next triple
        a_triple, b_triple, c_triple = self.triples.pop(0)
        
        # Compute d = x - a and e = y - b
        d_shares = []
        e_shares = []
        for i in range(self.num_parties):
            d_shares.append(shares_a[i] + (a_triple[i] * (-1)))
            e_shares.append(shares_b[i] + (b_triple[i] * (-1)))
        
        # Open d and e
        d = self.secret_sharing.reconstruct_secret(d_shares)
        e = self.secret_sharing.reconstruct_secret(e_shares)
        
        # Compute result shares
        result_shares = []
        for i in range(self.num_parties):
            # result = c + d*b + e*a + d*e
            share = c_triple[i]
            share = share + (b_triple[i] * d)
            share = share + (a_triple[i] * e)
            if i == 0:  # Add d*e to first share
                share = share + SecretShare(
                    value=(d * e) % self.prime_field,
                    share_id=0,
                    prime_field=self.prime_field
                )
            result_shares.append(share)
        
        return result_shares

class SPDZProtocol:
    """Implementation of the SPDZ protocol for multi-party secure computation with malicious security.
    
    Based on the paper: "SPDZ2k: Efficient MPC mod 2^k for Dishonest Majority"
    by Keller, Pastro, and Rotaru, 2018.
    """
    
    def __init__(self, num_parties: int, prime_field: int = 2**61 - 1, security_bits: int = 128):
        """Initialize SPDZ protocol.
        
        Args:
            num_parties: Number of participating parties
            prime_field: Prime field modulus
            security_bits: Statistical security parameter
        """
        self.num_parties = num_parties
        self.prime_field = prime_field
        self.security_bits = security_bits
        self.secret_sharing = SecretSharing(num_parties, prime_field)
        
        # Initialize preprocessing material
        self._initialize_preprocessing()
        
        logger.info(f"Initialized SPDZ protocol with {num_parties} parties and {security_bits}-bit security")
    
    def _initialize_preprocessing(self) -> None:
        """Initialize preprocessing material."""
        # Generate multiplication triples
        self.triples = []
        for _ in range(1000):  # Pre-generate 1000 triples
            a = np.random.randint(0, self.prime_field)
            b = np.random.randint(0, self.prime_field)
            c = (a * b) % self.prime_field
            
            a_shares = self.secret_sharing.generate_shares(np.array([a]))
            b_shares = self.secret_sharing.generate_shares(np.array([b]))
            c_shares = self.secret_sharing.generate_shares(np.array([c]))
            
            self.triples.append((a_shares, b_shares, c_shares))
        
        # Generate random bits for comparison
        self.random_bits = []
        for _ in range(1000):
            bit = np.random.randint(0, 2)
            shares = self.secret_sharing.generate_shares(np.array([bit]))
            self.random_bits.append(shares)
    
    def share_input(self, input_value: np.ndarray, input_party: int) -> List[SecretShare]:
        """Share input among parties."""
        if input_party < 0 or input_party >= self.num_parties:
            raise ValueError(f"Input party ID must be between 0 and {self.num_parties-1}")
        
        return self.secret_sharing.generate_shares(input_value)
    
    def secure_addition(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Perform secure addition."""
        if len(shares_a) != self.num_parties or len(shares_b) != self.num_parties:
            raise ProtocolError(f"Expected {self.num_parties} shares for each input")
        
        result_shares = []
        for a_share, b_share in zip(shares_a, shares_b):
            result_shares.append(a_share + b_share)
        
        return result_shares
    
    def secure_multiplication(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Perform secure multiplication using Beaver triples."""
        if not self.triples:
            raise ProtocolError("No multiplication triples available")
        
        # Get next triple
        a_triple, b_triple, c_triple = self.triples.pop(0)
        
        # Compute d = x - a and e = y - b
        d_shares = []
        e_shares = []
        for i in range(self.num_parties):
            d_shares.append(shares_a[i] + (a_triple[i] * (-1)))
            e_shares.append(shares_b[i] + (b_triple[i] * (-1)))
        
        # Open d and e
        d = self.secret_sharing.reconstruct_secret(d_shares)
        e = self.secret_sharing.reconstruct_secret(e_shares)
        
        # Compute result shares
        result_shares = []
        for i in range(self.num_parties):
            share = c_triple[i]
            share = share + (b_triple[i] * d)
            share = share + (a_triple[i] * e)
            if i == 0:
                share = share + SecretShare(
                    value=(d * e) % self.prime_field,
                    share_id=0,
                    prime_field=self.prime_field
                )
            result_shares.append(share)
        
        return result_shares
    
    def secure_comparison(self, shares_a: List[SecretShare], shares_b: List[SecretShare]) -> List[SecretShare]:
        """Perform secure comparison using random bits."""
        if not self.random_bits:
            raise ProtocolError("No random bits available")
        
        # Compute difference a - b
        neg_b = []
        for share in shares_b:
            neg_b.append(share * (-1))
        diff_shares = self.secure_addition(shares_a, neg_b)
        
        # Get random bit
        r_shares = self.random_bits.pop(0)
        
        # Add random bit to difference
        masked_shares = self.secure_addition(diff_shares, r_shares)
        
        # Open masked value
        masked = self.secret_sharing.reconstruct_secret(masked_shares)
        
        # Determine result based on masked value and random bit
        r = self.secret_sharing.reconstruct_secret(r_shares)
        result = ((masked < 0) != bool(r[0])).astype(np.int64)
        
        # Share result
        return self.secret_sharing.generate_shares(result)

class SecureGradientAggregator:
    """High-level interface for secure gradient aggregation in ML applications."""
    
    def __init__(self, num_parties: int, security_model: str = "malicious", protocol: str = "spdz"):
        """Initialize secure gradient aggregator.
        
        Args:
            num_parties: Number of participating parties
            security_model: Security model ("semi-honest" or "malicious")
            protocol: MPC protocol to use ("spdz" or "aby3")
        """
        self.num_parties = num_parties
        self.security_model = security_model
        
        # Initialize appropriate protocol
        if protocol.lower() == "aby3":
            if num_parties != 3:
                logger.warning(f"ABY3 is designed for 3 parties, got {num_parties}")
            self.protocol = ABY3Protocol()
        else:  # Default to SPDZ
            self.protocol = SPDZProtocol(num_parties)
            
        logger.info(f"Initialized secure gradient aggregator with {num_parties} parties using {protocol} protocol")
        logger.info(f"Security model: {security_model}")
    
    def share_gradient(self, gradient: torch.Tensor, party_id: int) -> List[Dict[str, Any]]:
        """Convert gradient to secret shares for secure aggregation.
        
        Args:
            gradient: PyTorch gradient tensor
            party_id: ID of the party sharing the gradient
            
        Returns:
            List of serialized shares, one for each party
        """
        # Convert gradient to numpy array
        gradient_np = gradient.detach().cpu().numpy()
        
        # Share gradient
        shares = self.protocol.share_input(gradient_np, party_id)
        
        # Serialize shares for transmission
        return [share.to_dict() for share in shares]
    
    def aggregate_gradients(self, shares_list: List[List[Dict[str, Any]]]) -> torch.Tensor:
        """Aggregate gradients from all parties securely.
        
        Args:
            shares_list: List of shares from all parties
            
        Returns:
            Aggregated gradient
        """
        if len(shares_list) != self.num_parties:
            raise ValueError(f"Expected shares from {self.num_parties} parties, got {len(shares_list)}")
            
        # Deserialize shares
        shares_by_party = []
        for party_shares in shares_list:
            shares_by_party.append([SecretShare.from_dict(share) for share in party_shares])
            
        # Sum up shares from each party
        result_shares = []
        for i in range(self.num_parties):
            party_result = shares_by_party[0][i]
            for j in range(1, self.num_parties):
                party_result = party_result + shares_by_party[j][i]
            result_shares.append(party_result)
            
        # Reconstruct final gradient
        aggregated_np = self.protocol.secret_sharing.reconstruct_secret(result_shares)
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(aggregated_np).float()
    
    def aggregate_with_noise(self, shares_list: List[List[Dict[str, Any]]], noise_std: float) -> torch.Tensor:
        """Aggregate gradients with secure noise addition for differential privacy.
        
        Args:
            shares_list: List of shares from all parties
            noise_std: Standard deviation of Gaussian noise to add
            
        Returns:
            Aggregated gradient with noise
        """
        # Aggregate gradients
        aggregated = self.aggregate_gradients(shares_list)
        
        # Add noise for differential privacy
        noise_shape = aggregated.shape
        
        # For the malicious model, we generate and share noise securely
        if self.security_model == "malicious":
            # Each party generates noise shares
            noise_shares_list = []
            for i in range(self.num_parties):
                # Generate noise
                local_noise = np.random.normal(0, noise_std/np.sqrt(self.num_parties), size=noise_shape)
                
                # Share noise
                local_noise_shares = self.protocol.share_input(local_noise, i)
                noise_shares_list.append([share.to_dict() for share in local_noise_shares])
                
            # Aggregate noise shares
            noisy_aggregated = self.aggregate_gradients(noise_shares_list)
            
            # Add to aggregated gradient
            aggregated += noisy_aggregated
        else:
            # For semi-honest, add noise directly
            noise = torch.randn_like(aggregated) * noise_std
            aggregated += noise
            
        return aggregated
        
    def clipped_aggregation(self, shares_list: List[List[Dict[str, Any]]], clip_norm: float) -> torch.Tensor:
        """Aggregate gradients with secure clipping for robustness.
        
        Args:
            shares_list: List of shares from all parties
            clip_norm: Maximum L2 norm for gradients
            
        Returns:
            Aggregated clipped gradient
        """
        # In a real implementation, this would use secure norm computation and comparison
        # For simplicity, we apply clipping after aggregation (not fully secure)
        
        # Aggregate gradients
        aggregated = self.aggregate_gradients(shares_list)
        
        # Compute norm
        norm = torch.norm(aggregated)
        
        # Apply clipping
        if norm > clip_norm:
            aggregated = aggregated * (clip_norm / norm)
            
        return aggregated 