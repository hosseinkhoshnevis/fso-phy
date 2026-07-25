"""Every knob of the simulation in one dataclass, with defaults that
close the SDA OCT reference link: 2.5 Gbps OOK-NRZ at 1553.33 nm over
5500 km, 2.5 W transmitted through a 15 urad FWHM beam into an 8 cm
receive aperture ahead of an EDFA preamplifier.

Set rx_photons_per_bit to bypass the geometry and drive the receiver
directly, the way an OSNR sweep bypasses launch power in a fibre sim.
"""

from dataclasses import dataclass

H_NU = 6.62607015e-34 * 193.0e12   # photon energy at channel -1 (J)


@dataclass
class SimConfig:
    # line format
    baud: float = 2.5e9            # symbol rate (chips/s on the wire)
    encoding: str = "nrz"          # "nrz" | "manchester"
    pl_rate: int = 4               # 0 uncoded, 1..4 per Table 3-11
    n_frames: int = 12             # first frame is acquisition, not counted
    # transmitter
    tx_power_w: float = 2.5        # at the aperture (OCT-035 floor)
    er_db: float = 30.0            # extinction ratio
    rin_db_hz: float = -145.0      # relative intensity noise
    tx_bw: float = 0.75            # 4th-order Bessel, fraction of baud
    tx_loss_db: float = 1.5        # optics after the power figure
    # geometry
    range_km: float = 5500.0
    div_fwhm_urad: float = 15.0    # transmit FWHM divergence (OCT-040 floor)
    rx_aperture_m: float = 0.08
    rx_loss_db: float = 2.0        # receive optics to the preamp input
    # pointing
    jitter_urad: float = 2.0       # 1-sigma per axis, radial error is Rayleigh
    jitter_bw_hz: float = 300.0    # Gauss-Markov bandwidth of the jitter
    bias_urad: float = 0.0         # static mispoint
    # Doppler / clocks
    clock_ppm: float = 25.0        # symbol-rate offset seen by the receiver
    # receiver front end
    preamp_gain_db: float = 35.0
    preamp_nf_db: float = 4.5
    b_opt_ghz: float = 20.0        # optical filter, two-sided
    pol_filter: bool = False       # True halves the ASE (single-pol receiver)
    rx_bw: float = 0.75            # electrical 4th-order Bessel, fraction of baud
    thermal_ph_s: float = 0.0      # input-referred thermal noise, photon-rate units
    background_ph_s: float = 0.0   # stray/solar background photon rate in B_opt
    adc_bits: int = 8
    adc_sps: int = 2
    # receiver DSP
    timing_block: int = 64         # symbols per Gardner loop update
    timing_kp: float = 0.08
    timing_ki: float = 1e-4
    ffe_taps: int = 7              # 0 disables the little post-equalizer
    ldpc_iters: int = 40
    ldpc_alg: str = "nms"          # "nms" | "spa" (exact tanh rule)
    # simulation
    sps: int = 8                   # waveform samples per chip
    rx_filter: str = "bessel"      # "bessel" | "id" (integrate-and-dump)
    rx_photons_per_bit: float = 0.0  # >0 overrides the link budget
    include_impairments: bool = True  # False: static clean channel for tests
