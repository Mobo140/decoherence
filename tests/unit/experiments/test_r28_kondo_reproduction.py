"""R28: helpers of the Kondo reproduction and its smoke run."""
import math

import numpy as np
import pytest

from experiments import r28_kondo_reproduction as r28
from src.physics import kondo_bath


def test_deviation_reports_the_largest_absolute_difference():
    record = r28._deviation("fig", "q", np.array([1.0, 2.0]),
                            np.array([1.5, 2.1]))
    assert record["max_abs_deviation"] == pytest.approx(0.5)
    assert record["n_points"] == 2


def test_korringa_quadrature_matches_the_bath_rate():
    for delta in (-0.5, 0.0, 0.3, 2.0):
        assert r28._korringa_integral(delta, 0.2) == pytest.approx(
            kondo_bath.bath_rate(delta, 0.2), rel=1e-8)


def test_ising_lambda_of_a_single_site_is_four_and_of_a_pair_follows_the_kernel():
    assert r28._ising_lambda(np.ones(1), 0.3, 3) == pytest.approx(4.0)
    pair = r28._ising_lambda(np.ones(2), 0.7, 1)
    assert pair == pytest.approx(8.0 + 8.0 * math.cos(0.7) ** 2)


def test_paper_cos_cos_form_equals_closed_form_c3():
    for n_sites in (2, 5, 9):
        assert r28._ising_cos_cos(0.2, n_sites, "fm") == pytest.approx(
            r28._paper_lambda_1d(0.2, n_sites, "fm"))


def test_sector_problem_has_n_plus_one_states_and_the_spin_wave_gap():
    hamiltonian, ops = r28._sector_problem(4, -1.0, -1.0)
    energies = np.linalg.eigvalsh(hamiltonian.full())
    assert hamiltonian.shape == (5, 5) and len(ops) == 4
    assert energies[1] - energies[0] == pytest.approx(2.0)


def test_dimer_part_agrees_with_the_closed_forms(tmp_path, monkeypatch):
    monkeypatch.setattr(r28, "OUT_DIR", tmp_path)
    records = r28.part_dimer(fast=True)
    assert records
    assert max(r["max_abs_deviation"] for r in records) < 1e-10
    assert (tmp_path / "fig2_dimer.png").exists()
    assert (tmp_path / "r28_dimer.csv").exists()


# --- Negative cases ---

def test_paper_lambda_1d_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="unknown kind"):
        r28._paper_lambda_1d(0.1, 4, "afm_odd")
