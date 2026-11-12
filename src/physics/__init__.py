"""Physics of the simulated systems.

Modules:
    lindblad_systems: Hamiltonians, jump operators and the rate gamma(t).
    kondo_bath: Rates and collapse operators of spins exchange-coupled to
        an electron gas (Bloch-Redfield, secular limit).
    simulate_qutip: Adapts the Lindblad systems to ``blocks.evolve`` and
        evaluates observables.
    observables: Pauli expectation values of density matrices.
    measures: l1 coherence, purity, concurrence and related measures.
    paulis: Pauli matrices as NumPy constants.
"""
