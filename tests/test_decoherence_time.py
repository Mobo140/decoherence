"""Seed stability of random initial states."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stability():
    """Test stability: same seed → same output."""
    from src.physics.lindblad_systems import sample_random_pure_state
    state1 = sample_random_pure_state(n_qubits=1, seed=42)
    state2 = sample_random_pure_state(n_qubits=1, seed=42)
    
    # States should be identical
    diff = np.abs((state1.full() - state2.full()))
    assert np.allclose(diff, 0.0), "Same seed should produce same state"
    
    print("✓ Seed stability test passed")


if __name__ == '__main__':
    print("Running tests...")
    test_stability()
    print("All tests passed!")
