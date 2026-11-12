# R28: Kondo bath, checked against Delgado and Fernandez-Rossier

Reproduced from the generic code `src/physics/kondo_bath.py` (rates from
`kondo_rates`, exact diagonalisation; nothing is typed in from the closed
forms except for the overlay lines). Script:
`python -m experiments.r28_kondo_reproduction [--part dimer|swave|ising|surf|demo] [--fast]`.
Maximum deviations are in `deviations_<part>.csv`; "closed form" means the
formula as printed in the paper. Parameters: rho_J = 0.1 unless stated;
the ratios below do not depend on it. Agreement to about 1e-15 is machine
precision, not a measurement.

Two papers: PRB = PRB 95, 075413 (2017); SS = Surf. Sci. 630, 337 (2014).

`kondo_collapse_operators` is a pure function of (H, spin operators,
positions, rho_J, kT, k_F, dim, tol). A later simulator of sequential
measurement records can call it unchanged; `KondoSimulator` is a thin hook
that calls it from `_prepare_system`.

The collapse operators reproduce the rates in the dynamics: the
Liouvillian built from `kondo_collapse_operators` has, in the eigenbasis
of H, population rates equal to `kondo_rates.rates` and coherence decay
rates equal to `kondo_rates.dephasing` (random 3-spin Heisenberg chains in
random fields, d = 1, 2, 3: <= 1e-14; test
`test_collapse_dynamics_reproduce_rates_and_dephasing`). So the checks
below on the rates also hold for what the `kondo` simulator integrates.

## PRB Fig. 2(b, c): S = 1/2 dimer, J_H = 5 kT (`fig2_dimer.png`)

Gamma(T -> S) / Gamma0 at B = 0 and Gamma(|1,0> -> |1,-1>) / Gamma0 at
g mu_B B = 10 kT, k_F d / pi in [0, 2], d = 1, 2, 3.

- Closed forms 1 - K_d (Eqs. 28, 31-32) and 1 + K_d (Eqs. 30, 33-34):
  max abs deviation 6.7e-16 over 201 points per curve, all three d.
- Extra check, not in the figures: |0,0> -> |1,-1> in the large field,
  Gamma0(B - J_H) (1 - K_d): 4.4e-16.
- Discrepancy in the paper: the printed 2D and 3D curves of Fig. 2(b)
  are half of Eqs. 31-32, (1 - J0^2) / 2 and (1 - sinc^2) / 2 (digitised
  from the PDF: deviation <= 0.005 over the whole curve; from Eqs. 31-32
  themselves 0.5). The 1D curve of panel (b) and all three curves of
  panel (c) agree with the equations (<= 0.004). The text describes the
  figure ("emits at a rate half the one of an individual spin"). The
  matrix elements give Eqs. 31-32 (1 - K_d, limit 1); my panel (b)
  follows them, so its 2D and 3D curves are twice the printed ones.

## PRB Fig. 3: S = 3/2 FM chain, D = J_H = -1 meV, 3D gas (`fig3_spin_waves.png`)

H on the sectors S_z = 3N/2 and 3N/2 - 1 only (N + 1 states, from the
sparse 4^N operators); the spin operators are projected on them, and
`kondo_rates` runs on that space. Rates are in units of the single-spin
S = 3/2 rate at the excitation energy of SW1 (2 meV), the normalisation
under which the paper's SW1 curve reaches N at k_F d = 0 and the SW2
plateaus separate; kT = 0.02 meV, so G = Delta.

- Eq. 37: Gamma_SW1(N) / Gamma_SW1(1) = N, deviation <= 8e-15 for
  N = 4, 5, 6, 8 (SW1 is exactly uniform and has Delta = 2 meV).
- SW2 does not decay at k_F d = 0 (deviation <= 5e-17); SW2 maxima
  N = 4: 1.83 at k_F d / pi = 0.375, N = 5: 2.01 at 0.275, N = 6: 2.23 at
  0.25, N = 8: 2.74 at 0.175; plateau at k_F d / pi = 2: 1.44, 1.29, 1.20,
  1.11. Read by eye the paper's N = 8 SW2 maximum is about 2.7, at about
  0.2; its other values I did not read off precisely.
- Rates against sum_{ll'} K_ll' C_l C_l' with the ED amplitudes (an
  independent evaluation of the same formula): <= 4.4e-15.
- Fig. 3(a, b): the N = 8 amplitudes are in `r28_swave_coefficients_n8.csv`
  and the figure; SW1 uniform (0.354), SW2 antisymmetric and monotone from
  -0.49 to +0.49, the shapes of the paper's panels (by eye).

## PRB Fig. 4: Ising chains, k_F d / pi = 0.01 (`fig4_ising_dephasing.png`)

Lambda^FM and Lambda^AFM from the pure-dephasing rate between the two
degenerate ground states, Lambda = 4 s^T K s (s_l = +-1; checked against
`kondo_rates` on the full 2^N space for N = 2..8, d = 1, 2, 3: deviation
<= 5.7e-14), evaluated for N = 1..200.

- Two conventions give different curves. Eq. 16 has e^{+i q r_n} on one
  state and e^{-i q r_n} on the other. Averaged over the Fermi surface it
  gives 2 sum s_l s_l' [K(k_F |r_l - r_l'|) + K(k_F |r_l + r_l'|)], which
  depends on where the origin is. Appendix B (Eqs. B2, B3, B6) has only
  r_n - r_n', and so has `kondo_bath`: 4 s^T K s, translation invariant.
- The printed figure is Eq. 16 with the sites at r_l = l d (l = 1..N), in
  all nine panels/curves: digitised from the PDF, median deviation
  0.005-0.03 on a scale 0-4. The 1D closed forms C3 and C5 are exactly
  this sum (<= 1.5e-11). My solid curves are the invariant form, the
  dashed ones Eq. 16; the dashed ones reproduce the printed figure
  (e.g. 3D FM 0.88 and AFM odd N 1.5 at N k_F d / pi = 2).
- Which is right: the invariant one, I believe. For one spin Eq. 16 gives
  2 [1 + K(2 k_F r_1)], which depends on the position of the spin, while
  Eqs. 38-39 give 4; Appendix B, from which Eq. 16 is derived, has no such
  term. The authors' code is not available, so this is my reading, not a
  confirmed erratum. Both conventions share the k_F d -> 0 limit 4 N^2
  (FM) and the 1D period pi; they differ by up to 1.1 in Lambda^FM / N^2
  and 0.53 in Lambda^AFM (`deviations_ising.csv`).
- Eq. C4 and the 2D/3D AFM sums as printed have the argument
  xi |l - l'| in both terms, so they are the invariant sum (<= 2e-11), not
  the printed figure. Following Eq. C1 the second term should be
  xi (l + l').
- Eq. C6 (odd N) as printed differs from the paper's own sum and does not
  tend to 4 for k_F d -> 0. The sum of Eq. C1 is
  1/2 [4 + (1 + cos((2N + 1) xi) / cos xi)^2] (<= 1.6e-12).

## SS Table 1 / Fig. 2 (`fig_surfsci_shift.png`, `fig_surfsci_korringa.png`)

- Eq. 15: G(Delta) = Delta / (1 - exp(-Delta / kT)) against numerical
  quadrature of f(e)(1 - f(e + Delta)) for Delta / kT in [-6, 12]: 2.6e-13.
- Eq. 14, spin 1/2: generic rate against pi (rho J)^2 G(Delta) / 8 for
  Delta / kT in [0.01, 100]: exact; linear in Delta at Delta >> kT, and
  Gamma / (pi (rho J)^2 kT / 8) = 1.005 at Delta / kT = 0.01 (Korringa
  limit, the 0.5% is the first term of G(Delta) = kT (1 + Delta / 2 kT)).
- Eq. 21 against an independent evaluation of the same sum: 2e-16.
  The generic code reproduces Eq. 22 (spin 1/2), but not Eqs. 23 and 24:
  - S = 1, D = -3.5, E = 1.5 meV: Eq. 21 gives
    Delta~ / Delta = 1 - (3/4)(rho J)^2 ln(2W / pi kT) (deviation from
    this form 3.3e-16); Eq. 24 prints 3/2, a factor 2. Max deviation
    between generic and Eq. 24 over rho J in [0, 0.3]: 0.59.
  - S = 3/2, D = -3.5 meV, E = 0 and 2 meV (Delta0 = 9.85 meV for E = 2,
    as in the text): Eq. 21 gives the same 3/4 coefficient, independent of
    E; Eq. 23 prints Lambda / 8 with Lambda = 3/2 (1 + E^2 / D^2), 3/16 at
    E = 0, a factor 4 (deviation 0.44 at rho J = 0.3, with the paper's
    Lambda 0.40 for E = 2; ED gives Lambda = 3/2 for both).
  Eq. 21 with Eq. 22 and the S = 1 counting "the contribution within the
  doublet is twice the one from the 0 level" are consistent with each other
  (3 c with c = (rho J)^2 ln / 4); the printed prefactors of Eqs. 23 and 24
  are not. Eq. 21 gives the coefficient 3/4 for any quadratic anisotropy
  and any spin (checked for S = 1, 3/2, 2, 5/2), and S = 1 by hand. The
  copy here is the preprint; the journal version was not checked.
  With W = 1 eV, kT = 0.1 meV (ln = 8.7) halving 10 meV (the experimental case in the text) needs
  rho J = 0.28 from the generic formula and 1.4 from Eq. 23.
- The rate prefactors differ: SS Eq. 14 has pi / 2, PRB Eq. 20 pi / 4.
  Probably a different convention for rho (per spin or summed over
  spin), not checked. Only the absolute scale depends on it; every ratio
  above does not. I used PRB Eq. 20, which the PRB figures need
  (Eqs. 22, 28).
- No Lamb shift is added to the dynamics.

## Demo: two qubits under the `kondo` simulator (`fig_demo_t2.png`)

rho J = 0.1, kT = 0.2, J = 0.8 (TFIM with unit transverse field), t_max =
1200, dt = 1, three random initial states per point, 17 values of
k_F d / pi in [0, 2], d = 1, 2, 3, XXZ and TFIM: 306 trajectories, each
simulated for T2 (computational-basis l1 coherence) and T2^E (energy basis).
Medians over the initial states of uncensored runs.

- XXZ T2^E shows large RKKY-like oscillations: median 483 at k_F d = 0
  down to 93 (factor 5.2); for d = 1 with period pi in k_F d (maximum at 0,
  1, 2, minimum near 0.5 and 1.5), for d = 2, 3 a monotone fall with a weak
  residual wiggle. All uncensored.
- XXZ T2 (computational basis): 276 to 640, noisy; 30% of all T2 runs
  (both systems) reach t_max without crossing and are left out of the
  medians, so the T2 panels (XXZ, TFIM) are biased and not comparable
  with T2^E.
- TFIM T2^E: modulation of only 1.2% (96.1 to 97.3), and the three
  dimensions agree at k_F d = 0 and have the same range. T2 (computational
  basis) is 51 to 60 and jumps between two values: the 1/e crossing of an
  oscillating curve, which is the poor conditioning the project plan
  describes. I did not look for parameters where the TFIM modulation is
  larger.
- With three initial states the medians carry the sampling noise of three
  states; no uncertainty is estimated.
- Not done: scaling by the existing gamma(t) profile (the Kondo rates carry
  their own scale, there is no natural gamma_ref); a follow-up.
