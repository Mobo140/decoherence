"""R26: classification windows end at fixed t_obs and only while the pair is entangled."""
from types import SimpleNamespace

import numpy as np

from experiments import r26_esd_occurrence as r26


def _traj(t_death, n=201, dt=0.1):
    times = np.arange(n) * dt
    names = ["sigma_x1", "corr_xx", "concurrence", "purity", "coherence_l1"]
    M = np.tile(times[:, None], (1, len(names)))           # every column = time
    return SimpleNamespace(times=times, t_decoh=t_death, feature_names=names,
                           feature_matrix=M, system_config=SimpleNamespace(dt=dt))


def test_windows_end_at_fixed_times_and_skip_dead_pairs():
    X, owner, tob = r26.windows([_traj(2.7), _traj(20.0)], "corr_conc")
    # dies at 2.7: only t_obs 2.0 and 2.5; never dies: all five
    assert list(owner) == [0, 0, 1, 1, 1, 1, 1]
    assert list(tob) == [2.0, 2.5] + r26.T_OBS
    last = X.reshape(len(X), r26.WINDOW, -1)[:, -1, 0]     # time of the last point
    np.testing.assert_allclose(last, np.array(tob) - 0.1)  # window ends one step before t_obs


def test_feature_sets_drop_the_right_columns():
    for feats, width in (("base", 3), ("corr", 4), ("corr_conc", 5)):
        X, _, _ = r26.windows([_traj(20.0)], feats)
        assert X.shape[1] == r26.WINDOW * width
