#!/usr/bin/env python3
"""BER against received photons per bit, the currency of a free-space
optical link budget. Adaptive framing: transition points get as
many frames as it takes to either collect --min-errors post-FEC errors
or hit --max-frames, so the soft cliff is measured through the low
decades rather than extrapolated.

    python examples/sensitivity_sweep.py
    python examples/sensitivity_sweep.py --start 10 --stop 60 --points 14
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from fsophy import SimConfig, run_link
from fsophy.metrics import theory_ber_preamp_ook


def run_point(ph, args):
    tot = {"pre_e": 0, "pre_b": 0, "sd_e": 0, "sd_b": 0, "hd_e": 0, "hd_b": 0}
    frames = 0
    seed = 100
    while frames < args.max_frames:
        r = run_link(SimConfig(rx_photons_per_bit=ph, n_frames=args.frames,
                               pl_rate=args.pl), seed=seed)
        seed += 1
        frames += r["n_frames_counted"]
        tot["pre_e"] += r["pre_fec_errors"]
        tot["pre_b"] += r["pre_fec_bits"]
        for k in ("sd", "hd"):
            tot[f"{k}_e"] += r["post_fec"][k]["errors"]
            tot[f"{k}_b"] += r["post_fec"][k]["bits"]
        if frames < args.min_frames:
            continue
        if tot["sd_e"] >= args.min_errors and tot["hd_e"] >= args.min_errors:
            break                              # both curves resolved
        if tot["sd_e"] == 0 and tot["hd_e"] == 0 and frames >= 4 * args.min_frames:
            break                              # above both cliffs: floor point
        # sd error-free while hd still fails: the interesting decade,
        # keep going to max_frames so the soft curve is measured deep
    return tot, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=10.0)
    ap.add_argument("--stop", type=float, default=60.0)
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--pl", type=int, default=4)
    ap.add_argument("--frames", type=int, default=12, help="frames per run")
    ap.add_argument("--min-frames", type=int, default=11)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--min-errors", type=int, default=8)
    ap.add_argument("--grid", default=None,
                    help="explicit comma-separated photons/bit list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.grid:
        grid = np.array([float(x) for x in args.grid.split(",")])
    else:
        grid = np.geomspace(args.start, args.stop, args.points)
    out = Path(args.out or Path(__file__).with_name("sensitivity_sweep.csv"))
    rows = []
    for ph in grid:
        tot, frames = run_point(ph, args)
        row = {
            "photons_per_bit": round(float(ph), 3),
            "pre_ber": tot["pre_e"] / max(tot["pre_b"], 1),
            "sd_errors": tot["sd_e"], "sd_bits": tot["sd_b"],
            "hd_errors": tot["hd_e"], "hd_bits": tot["hd_b"],
            "frames": frames,
            "theory_ber": theory_ber_preamp_ook(float(ph)),
        }
        rows.append(row)
        print(f"ph {ph:6.2f}: pre {row['pre_ber']:.3e}  "
              f"sd {tot['sd_e']}/{tot['sd_b']}  hd {tot['hd_e']}/{tot['hd_b']}  "
              f"({frames} frames)")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
