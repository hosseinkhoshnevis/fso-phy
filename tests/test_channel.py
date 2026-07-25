"""Geometry, budget and pointing statistics."""

import numpy as np

from fsophy import SimConfig
from fsophy.channel import link_budget, mean_pointing_gain, pointing_gains


def test_reference_link_budget():
    b = link_budget(SimConfig())
    # 2.5 W through 15 urad at 5500 km beats the standard's 25 uW/m2 floor
    assert b["irradiance_uw_m2"] > 25.0
    assert 1500 < b["photons_per_bit"] < 4000
    assert -35 < b["p_rx_dbm"] < -28


def test_budget_scales_with_range():
    near = link_budget(SimConfig(range_km=1000))
    far = link_budget(SimConfig(range_km=8000))
    ratio_db = 10 * np.log10(near["p_rx_ph_s"] / far["p_rx_ph_s"])
    assert abs(ratio_db - 20 * np.log10(8)) < 0.01   # 1/Z^2, and nothing else


def test_outage_floor_formula():
    from fsophy.channel import outage_probability
    p4 = outage_probability(SimConfig(jitter_urad=4.0), 40.0, 11.0)
    p2 = outage_probability(SimConfig(jitter_urad=2.0), 40.0, 11.0)
    assert 0.005 < p4 < 0.15
    assert p2 < p4                       # calmer pointing, fewer outages
    assert outage_probability(SimConfig(jitter_urad=0.0), 40.0, 11.0) == 0.0
    assert outage_probability(SimConfig(jitter_urad=3.0), 10.0, 11.0) == 1.0


def test_pointing_fade_statistics_match_the_beta_model():
    cfg = SimConfig(jitter_urad=3.0, jitter_bw_hz=5e4)   # decorrelated frames
    rng = np.random.default_rng(0)
    g = pointing_gains(cfg, 40000, 7.2e-6, rng)
    assert abs(g.mean() - mean_pointing_gain(cfg)) < 0.02
    assert g.max() <= 1.0 and g.min() > 0.0
