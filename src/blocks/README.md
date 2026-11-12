# `src/blocks`: building a simulation from blocks

A simulation is a pipeline of six stages. Each stage is a module of small
pure functions; a block is a function plus its parameters.

```
system -> channels -> initial state -> evolution -> observation -> target
 H, spins   (L, rate)       rho0          rho(t)     named series    T2, T2^E, ESD
```

Channels combine by list concatenation, rate modifiers map a list of
channels to a list of channels, and the target is computed from rho(t), so
changing the target or the observation does not need a new simulation.
Randomness comes from explicit `numpy.random.Generator`s. The only use of
NumPy's global generator is `initial.random_pure` and
`initial.parity_sector` called without a generator, which exists to
reproduce the published runs.

Every example below is self-contained and is run by
`tests/integration/test_blocks_readme.py`.

Contents: [system](#system) | [channels](#channels) | [initial](#initial) |
[evolve](#evolve) | [observe](#observe) | [target](#target) |
[recipe](#recipe) | [samplers](#samplers) | [runner](#runner) |
[cache](#cache) | [legacy](#legacy) | [Dataset](#dataset) |
[Complete example](#complete-example)

## system

`system.py`. `System(H, spin_ops, positions, n_qubits)` is a frozen
dataclass: `H` is a `qutip.Qobj` on the full space, `spin_ops` holds
`(Sx, Sy, Sz)` of every site on the full space, `positions` has one row
per site (one lattice unit apart), `n_qubits` is the number of sites.
It validates shapes and raises `ValueError`.

`single_qubit(omega) -> System`
: H = omega * sigma_z. `omega` must be finite.

`two_qubit(J, interaction) -> System`
: `interaction` is `"XXZ"` (J (XX + YY + ZZ)) or `"TFIM"`
  (J XX + Z1 + Z2). Raises `ValueError` for another name or a non-finite J.

```python
from src.blocks import system

one = system.single_qubit(omega=1.0)
two = system.two_qubit(J=0.8, interaction="XXZ")
print(two.H.shape, two.positions)   # (4, 4) [0. 1.]
```

## channels

`channels.py`. `Channel(operator, rate)` is a frozen dataclass: the jump
operator `L` and a rate that is a non-negative constant or a function of
time.

`jump(system, op_type, gamma) -> list[Channel]`
: One channel per qubit, in qubit order, with constant rate `gamma`.
  `op_type` is `"sigma_minus"` or `"sigma_z"`; one or two qubits only.

`bernstein_profile(channels, coeffs, t_max) -> list[Channel]`
: Multiplies every rate by the Bernstein polynomial gamma(t). With unit
  rates the new rate is gamma(t) itself.

`ou_modulation(channels, s, tau, t_max, rng, modulate_constant=True)`
: Multiplies the rates by a mean-one log-normal Ornstein-Uhlenbeck factor
  of strength `s` and correlation time `tau` (R21). The generator is
  consumed in a fixed order: the increments first, then, if `s` is None,
  `s` and `tau`. With `modulate_constant=False` constant rates are left
  alone, but the path is still drawn.

`kondo_bath(system, rho_J, kT, k_F, dim, spacing=1.0) -> list[Channel]`
: Davies channels of spins coupled to an electron gas (R28), from
  `kondo_bath.kondo_collapse_operators`. The kernel depends on
  `k_F * spacing * distance`. Rates are inside the operators.

`to_mesolve(channels) -> list`
: The collapse-operator list of `qutip.mesolve`: `sqrt(r) * L` for a
  constant rate, `[L, sqrt_coeff]` for a function. Channels that share a
  rate function share one coefficient function.

Modifiers compose: they apply to everything before them in the list.

```python
import numpy as np
from src.blocks import channels, system

two = system.two_qubit(0.8, "XXZ")
base = channels.jump(two, "sigma_minus", gamma=1.0)
profile = channels.bernstein_profile(base, coeffs=(0.2, 0.5, 0.1), t_max=10.0)
noisy = channels.ou_modulation(
    profile, s=0.3, tau=2.0, t_max=10.0, rng=np.random.default_rng(0))
print(noisy[0].rate(3.0))           # gamma(3) times the OU factor
bath = channels.kondo_bath(two, rho_J=0.1, kT=0.2, k_F=1.0, dim=2)
print(len(bath), bath[0].rate)      # collapse operators, rate 1.0
```

## initial

`initial.py`.

`random_pure(system, rng=None) -> qutip.Qobj`
: Haar-random pure state as a density matrix. Without `rng` it draws from
  NumPy's global generator (legacy path); with a `Generator` it is private.

`parity_sector(system, sector, rng=None, *, min_norm=1e-3, max_draws=100)`
: Two qubits only. Left projection of a random state onto the parity
  sector +1 or -1, normalised in the trace norm (the known defect of R6 is
  kept). Draws again if the projection is below `min_norm`; `RuntimeError`
  after `max_draws`.

```python
import numpy as np
from src.blocks import initial, system

two = system.two_qubit(0.8, "XXZ")
rho0 = initial.random_pure(two, np.random.default_rng(1))
sector = initial.parity_sector(two, +1, np.random.default_rng(1))
print(rho0.shape, round(rho0.tr().real, 6))
```

## evolve

`evolve.py`. The only module that calls `qutip.mesolve`.

`master_equation(system, channels, rho0, times) -> States`
: `States(times, rho)`; `rho` is a list of dense `complex128` arrays.
  Raises `ValueError` for non-increasing or non-finite `times` and for
  shapes that do not fit `system.H`.

`sequential_record(system, channels, rho0, times, basis, rng, keep_rho=False) -> Record`
: Measures at `times[0]`, then between consecutive times evolves with
  `master_equation` on the interval and measures again; each outcome is
  drawn from the Born probabilities and collapses the state. Only
  `basis="z"`. `Record(times, outcomes, rho_after)`: `outcomes` is an
  integer array of shape `(n_times, n_qubits)`, 0 meaning sigma_z = +1.

```python
import numpy as np
import qutip as qt
from src.blocks import channels, evolve, system

one = system.single_qubit(1.0)
decay = channels.jump(one, "sigma_minus", gamma=0.5)
rho0 = qt.fock_dm(2, 0)
times = np.linspace(0.0, 2.0, 5)
states = evolve.master_equation(one, decay, rho0, times)
print(states.rho[-1].real.diagonal())   # population 0 is exp(-0.5 * 2)
record = evolve.sequential_record(
    one, decay, rho0, times, "z", np.random.default_rng(0))
print(record.outcomes.T)
```

## observe

`observe.py`. `Observations(times, series)` holds named series of shape
`(n_times,)` in insertion order. Series are addressed by name.

Exact, from `States`: `pauli_means` (`sigma_x, sigma_y, sigma_z` for one
qubit; `sigma_x1 ... sigma_z2` for two), `coherence_l1`, `purity`,
`concurrence` (two qubits), `correlators` (`corr_xx ... corr_zz`, two
qubits).

`finite_shots(states, n_shots, settings, rng) -> Observations`
: Projective shots. `settings="local"` measures both qubits in the same
  basis x, y, z and gives the Pauli means. `"tomography"` measures all
  3^n product bases, adds the correlators for two qubits, and reconstructs
  rho by linear inversion to give `purity` and `coherence_l1`. One qubit
  always gives both. Two qubits with `"local"` have no `purity` or
  `coherence_l1`; `model_layout` then raises and says they need
  tomography. Each mean is unbiased with variance `(1 - m^2) / n_shots`.

`record_series(record) -> Observations`
: `outcome_q1, outcome_q2, ...` as floats.

Transforms of `Observations`:
`combine(*observations)` merges series (duplicate names are an error),
`select(obs, names)` keeps the named series in that order,
`gaussian_noise(obs, sigma, rng, names=None)` and
`ar1_noise(obs, sigma, phi, rng, names=None)` add noise to one trajectory,
`model_layout(obs)` moves `MODEL_TAIL = ("purity", "coherence_l1")` to the
end, the one column order the models still read by position.
`ar1_filter(eta, phi)` turns white noise into stationary AR(1) noise.

```python
import numpy as np
import qutip as qt
from src.blocks import channels, evolve, observe, system

one = system.single_qubit(1.0)
states = evolve.master_equation(
    one, channels.jump(one, "sigma_z", 0.3),
    qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit()),
    np.linspace(0.0, 4.0, 9))
exact = observe.combine(
    observe.pauli_means(states), observe.purity(states),
    observe.coherence_l1(states))
shots = observe.finite_shots(
    states, n_shots=200, settings="tomography", rng=np.random.default_rng(0))
noisy = observe.ar1_noise(exact, sigma=0.02, phi=0.5,
                          rng=np.random.default_rng(1))
print(list(observe.model_layout(shots).series))
```

## target

`target.py`. `Target(time, censored)` is the decoherence time: the first
crossing of a level by a curve computed from rho(t), linearly
interpolated, or the last time with `censored=True`. Thresholds are in
(0, 1).

`coherence_crossing(states, system, threshold, criterion="coherence")`
: `criterion` is `"coherence"` or `"concurrence"` (the level is
  `threshold` times the initial value) or `"purity"` (the decay from the
  initial purity towards 1/2^n has fallen by `threshold`).

`energy_basis_crossing(states, system, threshold)`
: l1 coherence in the eigenbasis of H (T2^E).

`sudden_death(states, system)`
: Entanglement sudden death of two qubits.

Helpers: `first_crossing`, `interpolate_crossing`, `relative_level`,
`sudden_death_time`, `energy_basis_coherence`.

```python
import math
import numpy as np
import qutip as qt
from src.blocks import channels, evolve, system, target

one = system.single_qubit(1.0)
states = evolve.master_equation(
    one, channels.jump(one, "sigma_z", 0.3),
    qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit()),
    np.linspace(0.0, 10.0, 201))
print(target.coherence_crossing(states, one, 1 / math.e))
```

## recipe

`recipe.py`. A recipe is data: frozen and hashable.

`block(fn, **params) -> Block`
: `fn` is a dotted name relative to `src.blocks`, e.g.
  `"channels.kondo_bath"`. It must resolve to a public function and
  `params` must match its signature. Values are numbers, strings, `None`,
  tuples (lists become tuples) or `Param`. Parameters the runner supplies
  (`system`, `channels`, `states`, `record`, `obs`, `rng`) are not given.
  The runner also supplies `t_max` to blocks that have that parameter.

`Param(name)`
: Replaced per trajectory by the value the sampler returned under `name`.

`Recipe(system, channels, initial, observe, target, t_max=20.0, dt=0.1, sampler=None, seeds="spawn", legacy_rng_offset=0, evolve=block("evolve.master_equation"))`
: Stage lists. Channel blocks whose first parameter is `system` produce
  channels; those whose first parameter is `channels` modify everything
  before them. Observe blocks take `states` (or `record`) and produce, or
  take `obs` and transform. `target` may be `None`. `t_max` and `dt` may be
  `Param`s; the grid is `linspace(0, t_max, round(t_max / dt) + 1)`.
  The whole recipe is validated at construction (`ValueError`).

```python
from src.blocks.recipe import Param, Recipe, block

recipe = Recipe(
    system=block("system.single_qubit", omega=Param("omega")),
    channels=(block("channels.jump", op_type="sigma_z", gamma=0.4),),
    initial=block("initial.random_pure"),
    observe=(block("observe.pauli_means"), block("observe.purity"),
             block("observe.coherence_l1")),
    target=block("target.coherence_crossing", threshold=0.37),
    t_max=5.0, dt=0.1,
    sampler=block("samplers.uniform", omega=(0.5, 1.5)))
print(hash(recipe) == hash(recipe), recipe.referenced_params())
```

## samplers

`samplers.py`. `sampler(rng, n, **params) -> list[dict]`: the parameter
values of `n` trajectories. Trajectory `i` does not depend on `n`.
`uniform(rng, n, **ranges)` draws from `(low, high)` ranges,
`choice(rng, n, **options)` from lists, `grid(rng, n, **axes)` walks the
Cartesian product and repeats, `fixed(rng, n, **values)` repeats,
`from_rows(rng, n, rows)` replays rows of `(name, value)` pairs.

```python
import numpy as np
from src.blocks import samplers

rng = np.random.default_rng(0)
print(samplers.uniform(rng, 2, J=(0.2, 1.0)))
print(samplers.grid(rng, 3, a=[1, 2], b=["x"]))
```

## runner

`runner.py`.

`run(recipe, n_trajectories, seed, cache_dir=None, keep_states=False, indices=None) -> list[Run]`
: `Run(index, params, system_params, system_fn, jump_op, observations,
  target, states, record)`. With `seeds="spawn"` trajectory `i` depends
  only on `(seed, i)` and its sampled parameters: the sampler uses
  `SeedSequence(seed, spawn_key=(0,))`, trajectory `i` uses
  `SeedSequence(seed, spawn_key=(1, i))` whose children drive the
  channels, initial, observe and evolve stages. `indices` computes a
  subset. With `seeds="legacy"` the global NumPy generator is seeded once,
  trajectories run in order, the initial state comes from the global
  generator and the other random blocks share
  `default_rng(seed + legacy_rng_offset)`.

`run_with_generator(recipe, rng, index=0, seed=0, params=None, cache_dir=None, keep_states=False) -> Run`
: One trajectory of a legacy recipe with a generator the caller owns, for
  simulator objects that persist between trajectories.

```python
from src.blocks.recipe import Param, Recipe, block
from src.blocks.runner import run

recipe = Recipe(
    system=block("system.single_qubit", omega=1.0),
    channels=(block("channels.jump", op_type="sigma_z", gamma=0.4),),
    initial=block("initial.random_pure"),
    observe=(block("observe.pauli_means"), block("observe.purity"),
             block("observe.coherence_l1")),
    target=block("target.coherence_crossing", threshold=0.37),
    t_max=5.0, dt=0.1)
runs = run(recipe, n_trajectories=3, seed=7)
again = run(recipe, n_trajectories=5, seed=7, indices=[2])
print(runs[2].target == again[0].target)   # True
```

## cache

`cache.py`. `run(..., cache_dir=...)` stores `States` on disk as
`complex128` `.npz` files named by a sha1 of the resolved system, channel
and initial blocks, the time grid, the seed identity and the bytes of the
integration input. A hit returns exactly the bits of a miss. The initial
state and the channels are built before the lookup, so the random
generators advance the same on a hit. `states_key`, `load`, `store` are the
building blocks. Applies to `evolve.master_equation` only.

```python
import tempfile
from src.blocks.recipe import Recipe, block
from src.blocks.runner import run

recipe = Recipe(
    system=block("system.single_qubit", omega=1.0),
    channels=(block("channels.jump", op_type="sigma_z", gamma=0.4),),
    initial=block("initial.random_pure"),
    observe=(block("observe.purity"),), target=None, t_max=2.0, dt=0.1)
with tempfile.TemporaryDirectory() as directory:
    first = run(recipe, 1, seed=3, cache_dir=directory)[0]
    second = run(recipe, 1, seed=3, cache_dir=directory)[0]
print((first.observations.series["purity"]
       == second.observations.series["purity"]).all())
```

## legacy

`legacy.py`. `recipe_for(config, alias="plain", **params) -> Recipe`
turns a `SystemConfig` and a simulator alias (`plain`, `correlators`,
`entanglement`, `stochastic_gamma`, `parity_sector`, `kondo`; arguments in
`ALIAS_PARAMS`) into a legacy-seeds recipe that gives the published numbers
bit for bit. `src.simulation.simulator.QuTipSimulator(alias, cache_dir,
**params)` is a facade over it with the `ISimulator` interface.

```python
from src.data.scenarios import build_configs
from src.simulation.simulator import QuTipSimulator

config = build_configs(1, 0)["B"][0]
trajectory = QuTipSimulator("plain").simulate(config)
print(trajectory.feature_names, round(trajectory.t_decoh, 2))
```

## Dataset

`src/data/recipe_dataset.py` (it depends on the dataset code, so it is not
in `src/blocks`).

`to_dataset(runs, *, window_length, horizon, samples_per_trajectory, train_ratio, val_ratio, min_window_gap, include_censored, rng) -> Dataset`
: Cuts runs into windows with `windows.build_dataset`. `Dataset.context`
  gets `J`, `interaction_type` and `dissipator_type` where they are
  defined; the dissipator codes need a `channels.jump` block. Raises
  `ValueError` for a run without a target.

## Complete example

Kondo bath with a stochastic rate on two qubits: run, then cut windows;
then the same system observed with a finite number of shots.

```python
import numpy as np
from src.blocks.recipe import Param, Recipe, block
from src.blocks.runner import run
from src.data.recipe_dataset import to_dataset

def recipe(observation):
    return Recipe(
        system=block("system.two_qubit", J=Param("J"), interaction="XXZ"),
        channels=(
            block("channels.kondo_bath", rho_J=0.1, kT=0.2, k_F=1.0, dim=2),
            block("channels.ou_modulation", s=Param("s"), tau=2.0)),
        initial=block("initial.random_pure"),
        observe=observation,
        target=block("target.coherence_crossing", threshold=0.37),
        t_max=60.0, dt=1.0,
        sampler=block("samplers.uniform", J=(0.2, 1.0), s=(0.1, 0.5)))

exact = (block("observe.pauli_means"), block("observe.purity"),
         block("observe.coherence_l1"))
shots = (block("observe.finite_shots", n_shots=500, settings="tomography"),)

windows = dict(
    window_length=10, horizon=2.0, samples_per_trajectory=2,
    train_ratio=0.6, val_ratio=0.2, min_window_gap=0, include_censored=True)
for observation in (exact, shots):
    runs = run(recipe(observation), n_trajectories=6, seed=42)
    dataset = to_dataset(runs, rng=np.random.default_rng(0), **windows)
    print(dataset.sequences.shape, sorted(dataset.context))
```

A new dissipation term is a function `(System, ...) -> list[Channel]` in
`channels.py`, a test, and a paragraph above; a new observation is a
function `(States, ...) -> Observations` in `observe.py`.
