#!/usr/bin/env python3
"""Link performance against range: the geometry drives the budget, the
budget drives the receiver. The standard's compliance point (25 uW/m^2
at 5500 km) is marked in the output.

    python examples/range_sweep.py --start 1000 --stop 9000 --points 9
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from fsophy import SimConfig, run_link
from fsophy.channel import link_budget
from fsophy.metrics import q_db_from_ber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=1000.0)
    ap.add_argument("--stop", type=float, default=9000.0)
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for z in np.geomspace(a.start, a.stop, a.points):
        cfg = SimConfig(range_km=float(z), n_frames=a.frames)
        r = run_link(cfg, seed=int(z))
        b = link_budget(cfg)
        rows.append({
            "range_km": round(float(z), 1),
            "irradiance_uw_m2": round(b["irradiance_uw_m2"], 2),
            "photons_per_bit": round(b["photons_per_bit"], 1),
            "pre_ber": r["pre_fec_ber"],
            "sd_errors": r["post_fec"]["sd"]["errors"],
            "sd_bits": r["post_fec"]["sd"]["bits"],
            "hdr_ok": r["header_ok"], "frames": r["n_frames_counted"],
        })
        q = q_db_from_ber(max(r["pre_fec_ber"], 1e-12))
        print(f"z {z:7.0f} km: {b['irradiance_uw_m2']:8.1f} uW/m^2  "
              f"{b['photons_per_bit']:7.0f} ph/bit  pre {r['pre_fec_ber']:.2e}  "
              f"sd {r['post_fec']['sd']['errors']}")
    out = Path(a.out or Path(__file__).with_name("range_sweep.csv"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
