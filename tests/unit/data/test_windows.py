"""Window cutting: trajectory-level splits, window ends and context codes."""
import numpy as np

from src.data import windows


def test_assign_splits_is_a_trajectory_level_partition():
    split = windows.assign_splits(20, 0.8, 0.1, np.random.default_rng(1))
    assert split.count("train") == 16 and split.count("val") == 2
    assert split.count("test") == 2
    assert split == windows.assign_splits(
        20, 0.8, 0.1, np.random.default_rng(1))


def test_sample_window_ends_respects_the_minimum_gap():
    ends = np.arange(20, 200)
    chosen = windows.sample_window_ends(ends, 5, 10, np.random.default_rng(2))
    assert len(chosen) == 5
    gaps = np.abs(np.subtract.outer(chosen, chosen))[np.triu_indices(5, 1)]
    assert gaps.min() >= 10
    times = np.arange(30) * 0.1
    valid = windows.valid_window_ends(times, 2.0, 5)
    assert valid.min() == 5 and times[valid].max() < 2.0


def test_pad_features_inserts_zeros_before_the_last_two_columns():
    short = np.arange(12, dtype=np.float32).reshape(3, 4)
    full = np.ones((3, 6), dtype=np.float32)
    out = windows.pad_features([short, full])
    assert out[0].shape == (3, 6)
    np.testing.assert_array_equal(out[0][:, :2], short[:, :2])
    assert not out[0][:, 2:4].any()
    np.testing.assert_array_equal(out[0][:, 4:], short[:, 2:])


def test_context_codes_distinguish_the_four_system_types(scenario_configs):
    g = {sc: [cfg] for sc, cfg in scenario_configs.items()}
    codes = {sc: windows.context_codes(g[sc][0]) for sc in "ABCDE"}
    assert [codes[sc][0] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert [codes[sc][1] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert codes["A"][2] == 0.0 and codes["D"][2] == g["D"][0].J
