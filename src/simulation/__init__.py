"""Simulators that implement the domain ISimulator port.

Modules:
    simulator: QuTipSimulator and the decoherence-time functions.
    variants: Simulators that record extra observables or modify the
        dissipation, used by individual experiments.
    registry: `build_simulator` builds one of them by string alias.
"""
