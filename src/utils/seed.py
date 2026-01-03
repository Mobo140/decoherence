"""Utilities for setting random seeds for reproducibility."""
import random
import numpy as np
import jax
import os

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Configure JAX to prevent multiprocessing issues
    # This helps avoid segmentation faults when using JAX with multiprocessing
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    # Disable Orbax async operations that can cause segmentation faults
    os.environ['ORBAX_ASYNC'] = 'false'
    
    # Set JAX random key
    jax.random.PRNGKey(seed)
