"""Compatibility alias for the scenario generators, now in src.data.scenarios.

Only paper/paper_2/figures/replicated_figures.py still imports from here
(and the paper sources are not edited); experiments import
`src.data.scenarios` directly.
"""
from src.data.scenarios import build_configs, trajectory_store
from src.data.scenarios import bernstein_coeffs as _bernstein_coeffs

__all__ = ["build_configs", "trajectory_store", "_bernstein_coeffs"]
