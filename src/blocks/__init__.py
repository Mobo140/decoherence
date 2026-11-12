"""Building blocks of a simulation: small functions that compose.

Modules:
    system: The Hamiltonian, spin operators and site positions.
    channels: Dissipation channels (operator and rate) and their modifiers.
    initial: Initial states.
    evolve: Integration of the master equation, the only QuTiP call.
"""
