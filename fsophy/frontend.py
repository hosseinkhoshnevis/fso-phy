"""Electro-optics on both ends.

Transmit: NRZ or Manchester chips, a finite extinction ratio, laser RIN,
and a 4th-order Bessel bandwidth limit. Receive: an EDFA preamplifier
whose ASE is the noise that sets sensitivity, a two-sided optical
filter, square-law photodetection (signal-ASE and ASE-ASE beat noise
appear on their own from |E+n|^2, nothing is bolted on), an electrical
Bessel front end, and an ADC that samples at two samples per symbol on
its own imperfect clock. Fields are carried in photon-rate units
(|E|^2 = photons/s) referred to the preamp input.
"""

import numpy as np


def bessel4(x, cutoff_hz, fs):
    """Zero-phase-origin analog 4th-order Bessel applied in the
    frequency domain with reflect padding (frames are not circular)."""
    n = x.shape[-1]
    pad = min(n, 4096)
    xp = np.concatenate([x[..., pad - 1::-1], x, x[..., :-pad - 1:-1]], axis=-1)
    f = np.fft.rfftfreq(xp.shape[-1], 1.0 / fs)
    # normalised lowpass prototype, -3 dB at 2.1139 rad/s
    s = 1j * 2 * np.pi * f / (2 * np.pi * cutoff_hz) * 2.1139
    h = 105.0 / (s ** 4 + 10 * s ** 3 + 45 * s ** 2 + 105 * s + 105)
    y = np.fft.irfft(np.fft.rfft(xp, axis=-1) * h, n=xp.shape[-1], axis=-1)
    return y[..., pad:pad + n]


def tx_waveform(chips, cfg, rng):
    """Line chips (n,) -> optical power waveform (n*sps,), unit mean."""
    r = 10 ** (cfg.er_db / 10)
    p1, p0 = 2 * r / (r + 1), 2 / (r + 1)
    levels = np.where(chips > 0, p1, p0).astype(np.float64)
    x = np.repeat(levels, cfg.sps)
    fs = cfg.baud * cfg.sps
    if cfg.tx_bw < 2.0:              # >= 2 means an ideal rectangular TX
        x = bessel4(x, cfg.tx_bw * cfg.baud, fs)
    if cfg.include_impairments and cfg.rin_db_hz > -300:
        sigma = np.sqrt(10 ** (cfg.rin_db_hz / 10) * fs / 2)
        x = x * (1.0 + sigma * rng.standard_normal(x.shape))
    return np.clip(x, 0.0, None)


def receiver(power, cfg, rng):
    """Optical power waveform (photons/s at the preamp input) -> the
    electrical waveform after detection and filtering, same rate."""
    fs = cfg.baud * cfg.sps
    nf = 10 ** (cfg.preamp_nf_db / 10)
    n_sp = nf / 2.0                              # input-referred, G >> 1
    b_opt = min(cfg.b_opt_ghz * 1e9, fs)
    sig = np.sqrt(np.maximum(power, 0.0))
    noise = (rng.standard_normal((2, power.size)) +
             1j * rng.standard_normal((2, power.size)))
    noise *= np.sqrt(n_sp * fs / 2.0)            # per-pol complex ASE in fs
    if b_opt < fs:                               # brickwall optical filter
        spec = np.fft.fft(noise, axis=-1)
        f = np.fft.fftfreq(power.size, 1.0 / fs)
        spec[:, np.abs(f) > b_opt / 2] = 0.0
        noise = np.fft.ifft(spec, axis=-1)
    if cfg.pol_filter:
        noise[1] = 0.0
    ex = sig + noise[0]
    current = (ex.real ** 2 + ex.imag ** 2 + np.abs(noise[1]) ** 2)
    if cfg.thermal_ph_s > 0:
        current = current + cfg.thermal_ph_s * rng.standard_normal(current.shape)
    if cfg.background_ph_s > 0:
        current = current + cfg.background_ph_s
    if cfg.rx_filter == "id":
        k = cfg.sps
        kernel = np.ones(k) / k
        return np.convolve(current, kernel, mode="same")
    if cfg.rx_bw >= 2.0:             # ideal wideband electrical path
        return current
    return bessel4(current, cfg.rx_bw * cfg.baud, fs)


def adc(wave, cfg, rng, ppm, tau0):
    """Sample the waveform at adc_sps on a clock that is `ppm` fast and
    starts at fractional offset tau0 (UI), then quantise. Cubic
    interpolation off the dense simulation grid."""
    fs = cfg.baud * cfg.sps
    step = cfg.sps / cfg.adc_sps / (1.0 + ppm * 1e-6)
    n_out = int((wave.size - 4) / step)
    t = tau0 * cfg.sps + 1.0 + step * np.arange(n_out)
    y = cubic_interp(wave, t)
    if cfg.adc_bits > 0:
        lo, hi = np.percentile(y, 0.1), np.percentile(y, 99.9)
        span = (hi - lo) * 1.1
        q = np.round((y - lo) / span * (2 ** cfg.adc_bits - 1))
        y = np.clip(q, 0, 2 ** cfg.adc_bits - 1) / (2 ** cfg.adc_bits - 1) * span + lo
    return y


def cubic_interp(x, t):
    """Catmull-Rom cubic interpolation of x at fractional indices t."""
    i = np.floor(t).astype(np.int64)
    mu = t - i
    i = np.clip(i, 1, x.size - 3)
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    a = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
    b = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c = 0.5 * (x1 - xm1)
    return ((a * mu + b) * mu + c) * mu + x0
