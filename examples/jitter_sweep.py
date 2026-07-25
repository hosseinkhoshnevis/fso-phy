#!/usr/bin/env python3
"""Pointing jitter against performance at fixed range. Fades are quasi-
static over a frame, so what jitter really costs is frame outages, not
a smeared average BER; both are reported.

    python examples/jitter_sweep.py --photons 40
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from fsophy import SimConfig, run_link
from fsophy.channel import mean_pointing_gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=40.0,
                    help="photons/bit at zero pointing error")
    ap.add_argument("--jitters", default="0,1,2,3,4,5,6")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for j in [float(x) for x in a.jitters.split(",")]:
        pre_e = pre_b = sd_fe = fr = 0
        for k in range(a.runs):
            cfg = SimConfig(rx_photons_per_bit=a.photons, jitter_urad=j,
                            n_frames=a.frames)
            r = run_link(cfg, seed=300 + k)
            pre_e += r["pre_fec_errors"]
            pre_b += r["pre_fec_bits"]
            sd_fe += r["post_fec"]["sd"]["frame_errors"]
            fr += r["post_fec"]["sd"]["frames"]
        cfg = SimConfig(jitter_urad=j)
        rows.append({
            "jitter_urad": j,
            "mean_gain_db": round(10 * np.log10(mean_pointing_gain(cfg)), 3),
            "pre_ber": pre_e / max(pre_b, 1),
            "sd_frame_errors": sd_fe, "frames": fr,
            "outage": sd_fe / max(fr, 1),
        })
        print(f"jitter {j:.1f} urad: mean gain {rows[-1]['mean_gain_db']:+.2f} dB  "
              f"pre {rows[-1]['pre_ber']:.2e}  outage {sd_fe}/{fr}")
    out = Path(a.out or Path(__file__).with_name("jitter_sweep.csv"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
