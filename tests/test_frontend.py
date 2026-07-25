"""Electro-optic models: power bookkeeping and noise calibration."""

import numpy as np

from fsophy import SimConfig
from fsophy.frontend import receiver, tx_waveform


def test_tx_power_is_conserved():
    rng = np.random.default_rng(0)
    cfg = SimConfig(include_impairments=False)
    bits = rng.integers(0, 2, 20000, dtype=np.uint8)
    p = tx_waveform(bits, cfg, rng)
    assert abs(p.mean() - 1.0) < 0.01              # unit mean by construction
    assert p.min() >= 0.0


def test_extinction_ratio_levels():
    rng = np.random.default_rng(1)
    cfg = SimConfig(include_impairments=False, er_db=13.0, tx_bw=2.5)
    bits = np.array([0] * 50 + [1] * 50, dtype=np.uint8)
    p = tx_waveform(bits, cfg, rng)
    r = p[60 * cfg.sps] / p[20 * cfg.sps]
    assert abs(10 * np.log10(r) - 13.0) < 0.3


def test_dark_input_ase_power():
    rng = np.random.default_rng(2)
    cfg = SimConfig(include_impairments=False, rx_bw=2.5)
    n = 200000
    w = receiver(np.zeros(n), cfg, rng)
    fs = cfg.baud * cfg.sps
    n_sp = 10 ** (cfg.preamp_nf_db / 10) / 2
    expected = 2 * n_sp * fs                        # two polarisations
    assert abs(w.mean() / expected - 1.0) < 0.02
