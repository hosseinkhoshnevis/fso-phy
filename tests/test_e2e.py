"""End to end: the reference link closes, every mode runs, and the
decoders sit in the right order."""

import numpy as np
import pytest

from fsophy import SimConfig, run_link


def test_reference_link_is_error_free():
    r = run_link(SimConfig(n_frames=8), seed=1)
    assert r["photons_per_bit"] > 1000            # comfortable margin
    assert r["pre_fec_errors"] == 0
    assert r["post_fec"]["sd"]["errors"] == 0
    assert r["header_ok"] == r["n_frames_counted"]


def test_soft_beats_hard_near_the_cliff():
    r = run_link(SimConfig(n_frames=12, rx_photons_per_bit=13), seed=21)
    sd, hd = r["post_fec"]["sd"], r["post_fec"]["hd"]
    assert r["pre_fec_ber"] > 0.05
    assert sd["frame_errors"] <= hd["frame_errors"]


@pytest.mark.parametrize("pl", [0, 1, 4])
def test_every_pl_rate_closes_with_margin(pl):
    r = run_link(SimConfig(n_frames=6, rx_photons_per_bit=400, pl_rate=pl), seed=5)
    assert r["post_fec"]["sd"]["errors"] == 0


def test_manchester_mode():
    r = run_link(SimConfig(n_frames=6, rx_photons_per_bit=300,
                           encoding="manchester"), seed=9)
    assert r["post_fec"]["sd"]["errors"] == 0
    assert r["header_ok"] == r["n_frames_counted"]


def test_pointing_fades_cost_errors_only_when_deep():
    calm = run_link(SimConfig(n_frames=8, rx_photons_per_bit=60,
                              jitter_urad=0.5), seed=3)
    stormy = run_link(SimConfig(n_frames=8, rx_photons_per_bit=60,
                                jitter_urad=6.0), seed=3)
    assert calm["pre_fec_ber"] <= stormy["pre_fec_ber"]
