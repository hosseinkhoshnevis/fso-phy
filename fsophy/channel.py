"""The vacuum channel: geometry, pointing, and clocks.

There is no medium between the terminals, so the free-space channel has
no dispersion, no turbulence and no polarisation drift worth modelling;
what it does have is spreading loss over thousands of kilometres, a
received power that breathes with the microradian pointing error of two
body-steered telescopes, and a symbol clock that arrives Doppler-shifted
because the terminals move at kilometres per second relative to each
other. Photon-rate units are used throughout (watts divided by h*nu), so
link budget and receiver statistics share one currency.
"""

import numpy as np

from .config import H_NU


def gaussian_w_theta(div_fwhm_urad):
    """1/e^2 half-angle divergence from the FWHM the standard quotes."""
    return div_fwhm_urad * 1e-6 / np.sqrt(2.0 * np.log(2.0))


def link_budget(cfg):
    """On-axis received signal power at the preamp input, photon/s,
    plus the numbers a link engineer would want to see."""
    theta_w = gaussian_w_theta(cfg.div_fwhm_urad)
    z = cfg.range_km * 1e3
    p_tx = cfg.tx_power_w * 10 ** (-cfg.tx_loss_db / 10)
    w_z = theta_w * z
    irradiance = 2.0 * p_tx / (np.pi * w_z ** 2)          # W/m^2 on axis
    area = np.pi * (cfg.rx_aperture_m / 2) ** 2
    p_rx_w = irradiance * area * 10 ** (-cfg.rx_loss_db / 10)
    rb = cfg.baud if cfg.encoding == "nrz" else cfg.baud / 2
    return {
        "theta_w_urad": theta_w * 1e6,
        "irradiance_uw_m2": irradiance * 1e6,
        "p_rx_dbm": 10 * np.log10(p_rx_w * 1e3),
        "p_rx_ph_s": p_rx_w / H_NU,
        "photons_per_bit": p_rx_w / H_NU / rb,
    }


def pointing_gains(cfg, n_frames, frame_t, rng):
    """Per-frame power gain from pointing error.

    Each axis is a Gauss-Markov process at the jitter bandwidth; at a
    few hundred hertz against microsecond frames the error is frozen
    within a frame, so fading arrives frame by frame. The gain is the
    Gaussian-beam factor exp(-2 theta^2 / theta_w^2).
    """
    theta_w = gaussian_w_theta(cfg.div_fwhm_urad) * 1e6    # urad
    if cfg.jitter_urad <= 0 and cfg.bias_urad == 0:
        return np.ones(n_frames)
    rho = np.exp(-2 * np.pi * cfg.jitter_bw_hz * frame_t)
    e = rng.standard_normal((n_frames, 2)) * cfg.jitter_urad
    for k in range(1, n_frames):
        e[k] = rho * e[k - 1] + np.sqrt(1 - rho ** 2) * e[k]
    e[:, 0] += cfg.bias_urad
    r2 = (e ** 2).sum(axis=1)
    return np.exp(-2.0 * r2 / theta_w ** 2)


def mean_pointing_gain(cfg):
    """Closed form E[h] for the Rayleigh-pointing beta model, the
    number the fade statistics test pins the simulation against."""
    theta_w = gaussian_w_theta(cfg.div_fwhm_urad) * 1e6
    if cfg.jitter_urad <= 0:
        return 1.0
    g2 = theta_w ** 2 / (4.0 * cfg.jitter_urad ** 2)
    return g2 / (g2 + 1.0)


def outage_probability(cfg, photons_per_bit, cliff_photons):
    """Closed-form frame outage floor from geometry alone.

    With Rayleigh radial error the pointing gain is beta distributed,
    P(h < x) = x^{gamma^2} with gamma = theta_w / 2 sigma_j, so the
    probability that a frame arrives below the FEC cliff is simply the
    required gain raised to gamma^2. The receiver can only do worse
    than this: a faded frame also stresses the timing loop and the
    level estimators, which is what the measured outage sits above the
    prediction by."""
    if cfg.jitter_urad <= 0:
        return 0.0 if photons_per_bit >= cliff_photons else 1.0
    theta_w = gaussian_w_theta(cfg.div_fwhm_urad) * 1e6
    g2 = theta_w ** 2 / (4.0 * cfg.jitter_urad ** 2)
    h_req = cliff_photons / photons_per_bit
    if h_req >= 1.0:
        return 1.0
    return float(h_req ** g2)
