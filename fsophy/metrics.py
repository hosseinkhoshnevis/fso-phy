"""Metrics and the textbook curves the simulation is validated against.

The reference receiver for a preamplified OOK link is integrate-and-
dump behind an optical filter of M = B_opt * T modes. Its decision
statistic is exactly chi-square: on a one, one noncentral mode carrying
the signal plus 2M - 1 central ASE modes (two polarisations); on a
zero, the same with the extinction-ratio leakage as the noncentrality.
theory_ber_preamp_ook builds those densities numerically and reads the
minimum-error threshold off them, no Gaussian shortcut, so the AWGN-
style unit test pins the whole receiver calibration the way the 16QAM
curve pinned the fibre package.
"""

import numpy as np

from .config import H_NU


def photons_per_bit_to_dbm(nb, bit_rate):
    return 10 * np.log10(nb * bit_rate * H_NU * 1e3)


def dbm_to_photons_per_bit(dbm, bit_rate):
    return 10 ** (dbm / 10) * 1e-3 / (H_NU * bit_rate)


def q_db_from_ber(ber):
    from math import sqrt
    ber = np.clip(ber, 1e-300, 0.5 - 1e-12)
    # inverse of Q(x) via Newton on the complementary error function
    x = np.sqrt(2) * _erfcinv(2 * ber)
    return 20 * np.log10(np.maximum(x, 1e-12))


def _erfcinv(y):
    x = np.where(y < 1, np.sqrt(-np.log(np.maximum(y, 1e-300))), 0.0)
    from math import erfc
    v = np.vectorize(erfc)
    for _ in range(40):
        f = v(x) - y
        df = -2 / np.sqrt(np.pi) * np.exp(-x ** 2)
        x = x - f / np.where(np.abs(df) > 1e-300, df, 1e-300)
    return x


def _log_i0(x):
    small = np.log(np.i0(np.minimum(x, 30.0)))
    large = x - 0.5 * np.log(2 * np.pi * np.maximum(x, 1e-12))
    return np.where(x < 30.0, small, large)


def _density(lam, n0, m_central, grid):
    """pdf of one noncentral mode (noncentrality lam, per-mode noise
    n0) plus m_central central modes, by FFT convolution on `grid`."""
    du = grid[1] - grid[0]
    if lam > 0:
        pdf_s = np.exp(-(grid + lam) / n0 + _log_i0(2 * np.sqrt(grid * lam) / n0)) / n0
    else:
        pdf_s = np.exp(-grid / n0) / n0
    if m_central > 0:
        from math import lgamma
        k = m_central
        pdf_c = np.exp((k - 1) * np.log(np.maximum(grid, 1e-300)) - grid / n0
                       - k * np.log(n0) - lgamma(k))
        n = 2 * grid.size
        conv = np.fft.irfft(np.fft.rfft(pdf_s, n) * np.fft.rfft(pdf_c, n), n)[:grid.size] * du
    else:
        conv = pdf_s
    conv = np.maximum(conv, 0.0)
    return conv / max(conv.sum() * du, 1e-300)


def mutual_information(photons_per_bit, nf_db=4.5, m_modes=8, er_db=30.0,
                       dual_pol=True, n_grid=1 << 14):
    """Soft-decision mutual information of the chi-square OOK channel,
    bits per channel symbol, equiprobable inputs. This is the ceiling
    any code on this receiver can reach, so the distance between a
    measured FEC cliff and the photons where the MI equals the code
    rate is the honest coding-plus-implementation gap."""
    n_sp = 10 ** (nf_db / 10) / 2.0
    r = 10 ** (er_db / 10)
    n1 = 2.0 * photons_per_bit * r / (r + 1)
    n0 = 2.0 * photons_per_bit / (r + 1)
    m_central = (2 * m_modes - 1) if dual_pol else (m_modes - 1)
    top = (n1 + n_sp * (m_central + 1)) * 6 + 10 * n_sp
    grid = np.linspace(0, top, n_grid)
    du = grid[1] - grid[0]
    p1 = _density(n1, n_sp, m_central, grid)
    p0 = _density(n0, n_sp, m_central, grid)
    mix = 0.5 * (p0 + p1)
    eps = 1e-300
    i = 0.0
    for p in (p0, p1):
        mask = p > 1e-30
        i += 0.5 * np.sum(p[mask] * np.log2((p[mask] + eps)
                                            / (mix[mask] + eps))) * du
    return float(i)


def capacity_photons(rate, nf_db=4.5, m_modes=8, er_db=30.0,
                     dual_pol=True):
    """Photons per channel bit at which the soft-decision MI equals
    the code rate: the capacity threshold of this receiver."""
    lo, hi = 0.5, 80.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if mutual_information(mid, nf_db, m_modes, er_db, dual_pol) < rate:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def theory_ber_preamp_ook(photons_per_bit, nf_db=4.5, m_modes=8, er_db=30.0,
                          dual_pol=True, n_grid=1 << 14):
    """Exact chi-square BER of the ideal preamplified OOK receiver with
    an optimised threshold, equiprobable bits."""
    n_sp = 10 ** (nf_db / 10) / 2.0
    r = 10 ** (er_db / 10)
    n1 = 2.0 * photons_per_bit * r / (r + 1)
    n0 = 2.0 * photons_per_bit / (r + 1)
    m_central = (2 * m_modes - 1) if dual_pol else (m_modes - 1)
    top = (n1 + n_sp * (m_central + 1)) * 6 + 10 * n_sp
    grid = np.linspace(0, top, n_grid)
    du = grid[1] - grid[0]
    p1 = _density(n1, n_sp, m_central, grid)
    p0 = _density(n0, n_sp, m_central, grid)
    c1 = np.cumsum(p1) * du          # P(U <= u | 1)
    c0 = np.cumsum(p0) * du
    ber = 0.5 * (c1 + (1.0 - c0))
    k = int(np.argmin(ber))
    return float(ber[k])
