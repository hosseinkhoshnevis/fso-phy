#!/usr/bin/env python3
"""Eye diagrams of the received electrical waveform, folded at the
symbol period the Gardner loop actually recovered. Exposes capture()
for the paper's figure script.

    python examples/eye_diagram.py --photons 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from fsophy import SimConfig
from fsophy import channel, frontend, rxdsp
from fsophy.framing import build_frames


def capture(cfg, seed=1, n_bits=12000):
    """Return (t, traces) of the receiver waveform folded at the
    recovered symbol timing: t in UI, traces (n, 2*sps+1)."""
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, (2, 8416), dtype=np.uint8)
    line, _ = build_frames(payload, cfg.pl_rate, cfg.encoding)
    chips = line.reshape(-1)
    p = frontend.tx_waveform(chips, cfg, rng)
    ph = cfg.rx_photons_per_bit or channel.link_budget(cfg)["photons_per_bit"]
    w = frontend.receiver(p * ph * cfg.baud, cfg, rng)
    x2 = frontend.adc(w, cfg, rng,
                      cfg.clock_ppm if cfg.include_impairments else 0.0, 0.13)
    _, ftr = rxdsp.gardner(x2, cfg)
    # fold the dense waveform on the recovered rate for a 2-UI eye
    rate = 1.0 + (np.mean(ftr[len(ftr) // 2:]) if ftr.size else 0.0)
    period = cfg.sps * rate
    n = min(n_bits, int(w.size / period) - 3)
    starts = (np.arange(1, n) * period).astype(np.float64)
    k = np.arange(2 * cfg.sps + 1)
    idx = starts[:, None] + k[None, :]
    tr = frontend.cubic_interp(w, idx.reshape(-1)).reshape(n - 1, k.size)
    t = (k - cfg.sps) / cfg.sps
    return t, tr / max(tr.mean(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=200.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ph, tag in ((axes[0], a.photons, "comfortable"),
                        (axes[1], 16.0, "near the soft-FEC cliff")):
        t, tr = capture(SimConfig(rx_photons_per_bit=ph))
        ax.plot(t, tr[:2500].T, color="#2a78d6", lw=0.3, alpha=0.05)
        ax.set_title(f"{ph:.0f} photons/bit ({tag})")
        ax.set_xlabel("time (UI)")
    axes[0].set_ylabel("detected power (a.u.)")
    out = a.out or Path(__file__).with_name("eye_diagram.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
