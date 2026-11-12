"""Building blocks of a simulation: small functions that compose.

Modules:
    system: The Hamiltonian, spin operators and site positions.
    channels: Dissipation channels (operator and rate) and their modifiers.
    initial: Initial states.
    evolve: Integration of the master equation, the only QuTiP call.
    observe: Named series (observations) computed from the states, noise.
    target: Decoherence times (T2, T2^E, ESD) computed from the states.
"""
