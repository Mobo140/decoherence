"""R23: two-body correlators complete the two-qubit state and leave the base features unchanged."""
import numpy as np

from experiments._config import build_configs
from experiments.r23_two_body_correlators import CORR, CorrelatorSimulator, base_features
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def _cfgs():
    g = build_configs(n_per_scenario=3, seed=0)
    return g["D"] + g["E"]


def test_pauli_expectations_reconstruct_the_purity():
    for cfg in _cfgs():
        np.random.seed(1)
        obs = CorrelatorSimulator().simulate(cfg).observables
        single = sum(obs[f"sigma_{a}{q}"] ** 2 for a in "xyz" for q in "12")
        two = sum(obs[name] ** 2 for name in CORR)
        np.testing.assert_allclose((1 + single + two) / 4, obs["purity"], atol=1e-8)


def test_correlators_sit_before_purity_and_coherence():
    np.random.seed(1)
    names = CorrelatorSimulator().simulate(_cfgs()[0]).feature_names
    assert names[-2:] == ["purity", "coherence_l1"]
    assert set(CORR) <= set(names[:-2])


def test_base_features_equal_the_plain_simulator():
    cmd = dict(configs=_cfgs(), window_length=10, horizon=1.0, samples_per_trajectory=3,
               seed=4, dataset_name="r23_test")
    plain = GenerateDatasetUseCase(QuTipSimulator(), store=None).execute(GenerateDatasetCommand(**cmd))
    full = GenerateDatasetUseCase(CorrelatorSimulator(), store=None).execute(GenerateDatasetCommand(**cmd))
    base = base_features(full)
    assert base.feature_names == plain.feature_names
    np.testing.assert_array_equal(base.sequences, plain.sequences)
    np.testing.assert_array_equal(base.remaining_time, plain.remaining_time)
