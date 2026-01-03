"""Setup script for quantum decoherence prediction project."""
from setuptools import setup, find_packages

setup(
    name='quantum-decoherence-prediction',
    version='0.1.0',
    description='Predicting quantum decoherence times using transformers',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'qutip>=5.0.0',
        'jax>=0.4.20',
        'jaxlib>=0.4.20',
        'flax>=0.7.5',
        'optax>=0.1.7',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',
        'pyyaml>=6.0',
        'tensorboard>=2.13.0',
        'wandb>=0.15.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'einops>=0.6.1',
    ],
    python_requires='>=3.11',
)
