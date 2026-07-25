"""Clock recovery: the loop must swallow the Doppler-plus-oscillator
budget of a LEO crosslink without post-FEC errors."""

import numpy as np
import pytest

from fsophy import SimConfig, run_link


@pytest.mark.parametrize("ppm", [0.0, 25.0, 40.0, -40.0])
def test_pull_in(ppm):
    r = run_link(SimConfig(n_frames=8, rx_photons_per_bit=80, clock_ppm=ppm),
                 seed=7)
    assert r["post_fec"]["sd"]["errors"] == 0
    assert r["header_ok"] == r["n_frames_counted"]


def test_rate_estimate_tracks_sign():
    r_pos = run_link(SimConfig(n_frames=8, rx_photons_per_bit=200, clock_ppm=30.0), seed=8)
    r_neg = run_link(SimConfig(n_frames=8, rx_photons_per_bit=200, clock_ppm=-30.0), seed=8)
    assert r_pos["clock_ppm_est"] > 5.0
    assert r_neg["clock_ppm_est"] < -5.0
