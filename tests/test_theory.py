"""The calibration anchor: the near-ideal chain against the exact
chi-square curve of the preamplified OOK receiver."""

import numpy as np

from fsophy import SimConfig, run_link
from fsophy.metrics import theory_ber_preamp_ook


def _cal_cfg(ph, frames=12):
    return SimConfig(n_frames=frames, include_impairments=False,
                     rx_photons_per_bit=ph, rx_filter="id", er_db=40,
                     rin_db_hz=-300, adc_bits=0, adc_sps=8, tx_bw=2.5,
                     ffe_taps=0)


def test_chain_matches_chi_square_theory():
    r = run_link(_cal_cfg(35), seed=11)
    th = theory_ber_preamp_ook(35, nf_db=4.5, m_modes=8, er_db=40)
    assert r["pre_fec_errors"] > 15                # statistically meaningful
    assert r["pre_fec_ber"] < 1.8 * th             # and close to exact theory
    assert r["pre_fec_ber"] > 0.6 * th


def test_theory_reproduces_the_textbook_quantum_limit():
    # the classic preamplified-OOK figure: ~38-40 photons/bit at 1e-9
    # for one noise mode, a 3 dB noise figure and a polarisation filter
    lo, hi = 10.0, 200.0
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if theory_ber_preamp_ook(mid, nf_db=3.0, m_modes=1, er_db=40,
                                 dual_pol=False) > 1e-9:
            lo = mid
        else:
            hi = mid
    assert 37.0 < (lo * hi) ** 0.5 < 41.0


def test_capacity_threshold_sits_below_the_measured_cliff():
    from fsophy.metrics import capacity_photons, mutual_information
    cap = capacity_photons(0.5, er_db=40)
    assert 6.0 < cap < 8.5                 # rate-1/2 on the M=8 receiver
    assert abs(mutual_information(cap, er_db=40) - 0.5) < 0.01
    assert capacity_photons(0.8462, er_db=40) > cap


def test_theory_curve_behaviour():
    a = theory_ber_preamp_ook(30)
    b = theory_ber_preamp_ook(60)
    assert b < a / 20                              # steeply monotone
    # finite extinction costs sensitivity
    assert theory_ber_preamp_ook(40, er_db=10) > 3 * theory_ber_preamp_ook(40, er_db=40)
    # a polarisation filter helps
    assert theory_ber_preamp_ook(40, dual_pol=False) < theory_ber_preamp_ook(40, dual_pol=True)
