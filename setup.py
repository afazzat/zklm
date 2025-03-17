from setuptools import setup, find_packages

setup(
    name="zkml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "cryptography>=41.0.0",
        "web3>=6.0.0",
        "py-ecc>=6.0.0",
        "flwr>=1.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
) 