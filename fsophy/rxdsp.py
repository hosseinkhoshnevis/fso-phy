"""Receiver DSP: timing recovery, frame sync, a small equalizer, level
estimation and soft demapping.

The chain is deliberately the direct-detection counterpart of a
coherent receiver's: the carrier problems are gone (there is no
carrier after square-law detection), and the clock problem takes their
place. The ADC free-runs tens of ppm from the transmit symbol clock
because of oscillator tolerance and Doppler, so a Gardner loop with a
cubic interpolator recovers symbol timing before anything else can
work. Everything runs block-vectorised: the loops update once per
block of symbols on block-averaged error terms, numpy inside.
"""

import numpy as np

from .frontend import cubic_interp
from .framing import PREAMBLE


def gardner(x2, cfg):
    """Timing recovery on the 2 samples/symbol ADC stream.

    Per block of `timing_block` symbols the loop interpolates on-time
    and mid samples at the current phase and rate, averages the Gardner
    error e[k] = mid_k * (on_k - on_{k-1}) over the block, and updates a
    PI pair: the integrator absorbs the ppm-scale rate offset, the
    proportional path the residual phase. Returns the symbol-rate
    samples and the loop's rate estimate history.
    """
    x = x2 - x2.mean()
    nb = cfg.timing_block
    step0 = float(cfg.adc_sps)
    f = 0.0                                   # fractional rate correction
    p = 2.0                                   # read pointer, input samples
    out = []
    f_trace = []
    scale = 1.0 / max(x.std(), 1e-12) ** 2
    while p + (nb + 2) * step0 * (1 + abs(f)) < x.size - 4:
        step = step0 * (1.0 + f)
        t_on = p + step * np.arange(nb)
        y_on = cubic_interp(x, t_on)
        y_mid = cubic_interp(x, t_on - step / 2.0)
        e = np.mean(y_mid[1:] * (y_on[1:] - y_on[:-1])) * scale
        f -= cfg.timing_ki * e
        f = np.clip(f, -8e-5, 8e-5)               # +/- 80 ppm capture range
        p += nb * step - cfg.timing_kp * e * step0
        out.append(y_on)
        f_trace.append(f)
    y = np.concatenate(out) if out else np.zeros(0)
    return y + x2.mean(), np.array(f_trace)


def frame_sync(y, n_frame, ref_bits=None):
    """Find the first frame start by correlating against the bipolar
    preamble (in whatever chip pattern is on the wire) over a
    two-frame search window."""
    if ref_bits is None:
        ref_bits = PREAMBLE
    ref = 2.0 * np.asarray(ref_bits, dtype=np.float64) - 1.0
    m = ref.size
    z = y - y.mean()
    n = min(2 * n_frame + m, z.size - m)
    c = np.array([z[k:k + m] @ ref for k in range(n)])
    return int(np.argmax(c))


def estimate_levels(y, iters=6):
    """Blind two-level clustering: means, sigmas and the crossing
    threshold. OOK noise is level-dependent (signal-ASE beating on the
    ones), so both sigmas matter, not just the midpoint."""
    mu0, mu1 = np.percentile(y, 25.0), np.percentile(y, 75.0)
    for _ in range(iters):
        th = 0.5 * (mu0 + mu1)
        lo, hi = y[y < th], y[y >= th]
        if lo.size == 0 or hi.size == 0:
            break
        mu0, mu1 = lo.mean(), hi.mean()
    th = 0.5 * (mu0 + mu1)
    s0 = max(y[y < th].std(), 1e-9)
    s1 = max(y[y >= th].std(), 1e-9)
    # move the threshold to the likelihood crossing of the asymmetric pair
    for _ in range(8):
        th = (mu0 * s1 + mu1 * s0) / (s0 + s1)
        s0 = max(y[y < th].std(), 1e-9)
        s1 = max(y[y >= th].std(), 1e-9)
    return mu0, mu1, s0, s1


def llrs(y, mu0, mu1, s0, s1):
    """Per-symbol LLR log p(y|0)/p(y|1) with each level modelled as a
    Gamma density matched to its first two moments.

    Beat-noise statistics are chi-square, not Gaussian: the ones are
    noisier than the zeros and both are right-skewed. A Gaussian
    crossing puts the threshold visibly too high and was measured to
    cost about a factor three in pre-FEC BER; the Gamma pair lands
    within counting noise of the sweep-optimal threshold and hands the
    decoder honestly shaped soft information."""
    from math import lgamma
    k0, t0 = (mu0 / s0) ** 2, s0 ** 2 / mu0
    k1, t1 = (mu1 / s1) ** 2, s1 ** 2 / mu1
    if not (np.isfinite(k0) and np.isfinite(k1)) or mu0 <= 0 or k0 > 5e3:
        l = (np.log(s1 / s0) + (y - mu1) ** 2 / (2 * s1 ** 2)
             - (y - mu0) ** 2 / (2 * s0 ** 2))
        return np.clip(l, -40.0, 40.0)
    u = np.maximum(y, 1e-4 * mu1)
    lu = np.log(u)
    l = ((k0 - 1) * lu - u / t0 - k0 * np.log(t0) - lgamma(k0)
         - (k1 - 1) * lu + u / t1 + k1 * np.log(t1) + lgamma(k1))
    return np.clip(l, -40.0, 40.0)


def ffe(y, mu0, mu1, taps, sweeps=2, block=512, mu=0.05):
    """Little T-spaced decision-directed feed-forward equalizer that
    cleans up the residual ISI of the two Bessel front ends. Block-LMS
    with the centre tap seeded, decisions against the two levels."""
    if taps <= 1:
        return y
    span = (mu1 - mu0) or 1.0
    x = (y - mu0) / span * 2.0 - 1.0          # roughly bipolar
    w = np.zeros(taps)
    w[taps // 2] = 1.0
    half = taps // 2
    xp = np.pad(x, half, mode="edge")
    cols = np.lib.stride_tricks.sliding_window_view(xp, taps)[: x.size]
    for _ in range(sweeps):
        for a in range(0, x.size - block, block):
            seg = cols[a:a + block]
            z = seg @ w[::-1]
            d = np.where(z > 0, 1.0, -1.0)
            err = d - z
            w[::-1] += mu * (err[:, None] * seg).mean(axis=0)
    z = cols @ w[::-1]
    return (z + 1.0) / 2.0 * span + mu0
