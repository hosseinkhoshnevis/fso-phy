# fsophy

**Hossein Khoshnevis** · MIT licence

Baseband PHY simulation of a free-space optical link modem in plain
vectorised numpy, built to the SDA Optical Communications Terminal
(OCT) Standard v3.2.0, whose home is the inter-satellite crosslink: 2.5 Gbps OOK-NRZ at 1553.33 nm, the
standard's modem frame, its 5G NR LDPC payload FEC, a preamplified
direct-detection receiver with full clock recovery, and pre- and
post-FEC BER measurement. It is the free-space sibling of
[ofm-phy](https://github.com/), the 800G coherent fibre modem
simulation, and follows the same discipline: bit-exact standard
structures, a receiver that earns its numbers blind, and validation
against textbook theory on one side and the standard on the other.

## What is inside

* **The OCT modem frame to the letter of the standard**: the 64-bit
  preamble `0x53225b1d0d73df03` MSB-first, the 960-bit header (128
  field bits, CRC-16, zero tail, through the non-systematic rate-1/6
  constraint-length-7 convolutional code with generators octal
  175/171/151/133/127/117), the frame scrambler from
  x^15 + x^14 + 1 restarted each frame from the standard seed, and
  back-to-back synchronous frames.
* **The payload FEC is the 5G NR LDPC** the standard adopts: base
  graph 1, lifting factor Z = 384, K = 8448 (the 8416 payload bits
  plus CRC-32 exactly fill the info block), first 768 systematic bits
  punctured, and the four PL_RATE modes transmitting 2304/3456/4992/
  9216 parity bits for rates 0.8462/0.7586/0.6667/0.5000, verified
  bit-for-bit against Table 3-11. Two decoders, since the standard
  fixes only the code: normalised min-sum on calibrated soft LLRs, and
  the same iteration on sign-only inputs, which measures what soft
  demapping is worth (about 2 dB at the cliff). The exact sum-product
  rule ships as an option (`ldpc_alg="spa"`) and was measured to gain
  nothing here, losing slightly at the cliff: the Gamma LLRs are
  approximate and min-sum's attenuation forgives that, exact belief
  propagation does not.
* **A vacuum channel with the impairments that actually matter in
  space**: 1/Z^2 spreading with the Gaussian-beam geometry (the
  standard's 2.5 W / 15 urad / 25 uW/m^2-at-5500-km numbers), frame-
  by-frame pointing fades from Gauss-Markov microradian jitter, and a
  symbol clock tens of ppm off from Doppler and oscillator tolerance.
  No dispersion, no turbulence, no polarisation drift: what vacuum
  does not do is modelled by leaving it out.
* **A preamplified receiver simulated at field level**: EDFA gain and
  noise figure, two-polarisation ASE, square-law detection so
  signal-ASE and ASE-ASE beat noise appear on their own, finite
  extinction ratio, RIN, Bessel electrical front ends, and an 8-bit
  ADC at 2 samples per symbol on its own clock.
* **Receiver DSP that earns the link blind**: Gardner timing recovery
  with a cubic interpolator (block-vectorised PI loop, +-80 ppm
  capture), preamble correlation frame sync, header Viterbi + CRC
  anchoring frame identity, per-frame two-level clustering that tracks
  pointing fades, Gamma-likelihood LLRs matched to the skewed beat-
  noise statistics (a Gaussian threshold was measured to cost 3x in
  BER and is kept as the fallback), and an optional little
  decision-directed FFE.
* **Measurement discipline**: pre-FEC BER on the line bits, post-FEC
  on the 8416-bit payloads, per-frame CRC-32 verdicts, an acquisition
  frame excluded from every count, and a chi-square theory anchor: in
  its ideal configuration the chain lands within ~0.2 dB of the exact
  noncentral-chi-square curve of the preamplified OOK receiver, and
  the FEC cliff within 0.9 dB of the channel's soft-decision capacity.

Everything is numpy. A 12-frame run of the full chain, both decoders
included, takes well under a second on a laptop-class core.

## Install

```
pip install -e .          # numpy only
pip install -e .[dev]     # + matplotlib and pytest
```

## Quick start

```python
from fsophy import SimConfig, run_link

r = run_link(SimConfig())            # the 5500 km reference link
print(r["budget"]["irradiance_uw_m2"])   # ~229, floor in the standard is 25
print(r["pre_fec_ber"], r["post_fec"]["sd"]["errors"])

r = run_link(SimConfig(rx_photons_per_bit=12))   # at the soft cliff
```

Command line:

```
python examples/run_single.py --range-km 7500 --jitter 3
python examples/sensitivity_sweep.py --start 10 --stop 60
python examples/range_sweep.py
python examples/jitter_sweep.py --photons 40
python examples/eye_diagram.py
```

Or use the FEC on its own:

```python
import numpy as np
from fsophy.fec import ldpc

u = np.random.default_rng(1).integers(0, 2, (4, 8448), dtype=np.uint8)
tx = ldpc.rate_match(ldpc.encode(u), 4)          # PL_RATE 4, rate 1/2
info, ok = ldpc.Decoder(4).decode(llrs_from_your_channel)
```

## Measured sensitivity

With every impairment on (Bessel front ends, ER 30 dB, RIN, 8-bit
2 sps ADC, 25 ppm clock offset, 2 urad pointing jitter), PL_RATE 4:

| photons/bit | pre-FEC BER | post-FEC (soft) | post-FEC (hard) |
|-------------|-------------|-----------------|-----------------|
| 10.25       | 1.5e-1      | 1.7e-1          | 2.2e-1          |
| 11.5        | 1.3e-1      | 3.7e-2          | 1.9e-1          |
| 13.5        | 9.6e-2      | 4.1e-3          | 8.7e-2          |
| 16          | 6.9e-2      | 8.1e-4 (outage tail) | 1.0e-2     |
| 18          | 5.1e-2      | 0 (< 3e-6)      | 0 (< 3e-6)      |

(Regenerate with `examples/sensitivity_sweep.py`; the committed CSV in
`docs/figures/` carries the exact numbers and depths.) The chain of
numbers is anchored on capacity, and it is checked, not asserted: the
soft-decision mutual information of the measured channel puts the
rate-1/2 capacity threshold at 7.2 photons per line bit
(`metrics.capacity_photons`); the measured cliff of the full blind
DSP on an ideal front end is 8.8, a 0.9 dB gap that is what
normalised min-sum gives up on this code; the default electro-optics
add 1.0 dB, putting the cliff at 11 photons per line bit (-54.5 dBm
at 2.5 Gbps, or 23 photons per information bit at the frame's 47%
payload fraction); and the default 2 urad of pointing jitter
stretches a thin outage tail out to about 17 before the error-free
floor. The theory engine itself reproduces the textbook 38.4
photons/bit quantum limit of preamplified OOK at one mode, NF 3 dB
and a polarisation filter, and a unit test holds it there. The
hard-input decoder needs roughly 5 photons more. Two more bounds:
headers and timing hold to about 7 photons per line bit, 2 dB beyond
the payload cliff, so the code sets the sensitivity rather than the
frame machinery; and Manchester pays about 2 dB against NRZ at the
same chip rate (cliff near 17), the price of a guaranteed transition
every bit. Fade outages have a closed form too,
`channel.outage_probability`: with Rayleigh pointing the frame-outage
floor is (cliff/photons)^(gamma^2), and the measured outage rides
above it because a deep fade also drags the timing loop and the level
estimators. The
reference geometry delivers ~2300 photons/bit at 5500 km, so the
standard's compliance point closes with double-digit dB of margin, and
the standard's 25 uW/m^2 floor corresponds to ~390 photons/bit at an
8 cm aperture, comfortably above the cliff at every PL_RATE.

## Layout

```
fsophy/
  config.py        SimConfig, every knob, OCT reference defaults
  sim.py           run_link: bits -> frames -> vacuum -> DSP -> BER
  fec/
    bg1.py         the 3GPP BG1 exponent table (iLS=1, Z=384)
    ldpc.py        encoder, rate matching, min-sum + hard-input decoders
    conv.py        header CC(7, 1/6) encoder + batch Viterbi
    crc.py         CRC-16 CCITT and CRC-32, zero-initialised
  framing.py       preamble, header, scrambler, frame build/parse
  channel.py       link budget, pointing fades, mean-gain closed form
  frontend.py      TX ER/RIN/Bessel, EDFA + ASE + square law, ADC
  rxdsp.py         Gardner + Farrow, frame sync, levels, Gamma LLRs, FFE
  metrics.py       photon/dBm conversions, exact chi-square OOK theory
examples/          run_single, sensitivity/range/jitter sweeps, eyes
tests/             41 tests: standard structure, decoder cliffs, theory
                   and quantum-limit match, capacity threshold, timing
                   pull-in, fade and outage statistics, end to end
docs/              measured data (CSV) and the paper (docs/paper/)
```

## Simulation conventions worth knowing

Fields are carried in photon-rate units referred to the preamp input,
so |E|^2 is photons per second and the link budget, the receiver
statistics and the theory curves share one currency. The optical
filter equals the simulation bandwidth at the default 8 samples per
chip, giving M = 8 noise modes per bit; the theory anchor uses the
same M. Frames are simulated as one continuous synchronous stream; the
first frame is acquisition and is excluded from every count. Frame
identity comes from decoded headers (CRC-16 anchored), not from a
genie. The 160-bit header field map models frame count, PL_RATE and
encoding; the standard's full Table 3-7 field allocation is richer,
and the header FEC structure around it is exact. Not modelled: the
PAT spiral-scan acquisition state machine, the 40/50 kHz AM tracking
tone, optical centre-frequency offsets (direct detection does not see
them), and inter-channel effects. The natural extension points are
noted in the module docstrings.

## References

* SDA Optical Communications Terminal (OCT) Standard v3.2.0 (sda.mil),
  Apr. 2025: frame, FEC modes, PHY floors
* 3GPP TS 38.212: the 5G NR LDPC (base graph 1) the standard adopts
* CCSDS 141.0-B-1 Optical Communications Physical Layer, for context
* P. A. Humblet and M. Azizoglu, "On the bit error rate of lightwave
  systems with optical amplifiers," JLT 1991: the chi-square anchor
* F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled
  receivers," IEEE Trans. Commun. 1986

MIT licence. If you spot a deviation from the standard, open an issue;
the structure tests in `tests/` are the place to encode it.
