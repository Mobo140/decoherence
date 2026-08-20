"""Test decoherence time calculation."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.decoherence_time import coherence_l1, purity, calculate_decoherence_time


def test_decoherence_time():
    """Test decoherence time calculation on synthetic trajectory."""
    # Create synthetic trajectory: start with coherent state, decay to diagonal
    n_times = 100
    times = np.linspace(0, 10, n_times)
    
    # Create density matrices: initial coherent state decaying to |0><0|
    density_matrices = []
    for i, t in enumerate(times):
        # Exponential decay of off-diagonal elements
        decay = np.exp(-t)
        rho = np.array([
            [1.0 - 0.5 * (1 - decay), 0.5 * decay],
            [0.5 * decay, 0.5 * (1 - decay)]
        ], dtype=complex)
        density_matrices.append(rho)
    
    # Calculate decoherence time using coherence criterion
    t_decoh = calculate_decoherence_time(
        times,
        density_matrices,
        criterion='coherence',
        threshold=0.01
    )
    
    assert t_decoh is not None, "Decoherence time should be found"
    assert t_decoh > 0, "Decoherence time should be positive"
    assert t_decoh <= times[-1], "Decoherence time should be within simulation time"
    
    print(f"✓ Decoherence time calculation test passed: t_decoh = {t_decoh:.4f}")


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
    test_decoherence_time()
    test_stability()
    print("All tests passed!")
