#!/usr/bin/env python3
"""One link, all numbers.

    python examples/run_single.py
    python examples/run_single.py --photons 20 --pl 4 --frames 12
    python examples/run_single.py --range-km 7500 --jitter 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fsophy import SimConfig, run_link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=0.0,
                    help="receiver photons/bit (0 = use the link budget)")
    ap.add_argument("--pl", type=int, default=4, help="PL_RATE 0..4")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--range-km", type=float, default=5500.0)
    ap.add_argument("--jitter", type=float, default=2.0, help="urad rms/axis")
    ap.add_argument("--ppm", type=float, default=25.0, help="clock offset")
    ap.add_argument("--manchester", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    cfg = SimConfig(rx_photons_per_bit=a.photons, pl_rate=a.pl,
                    n_frames=a.frames, range_km=a.range_km,
                    jitter_urad=a.jitter, clock_ppm=a.ppm,
                    encoding="manchester" if a.manchester else "nrz")
    r = run_link(cfg, seed=a.seed)
    b = r["budget"]

    print(f"range {cfg.range_km:.0f} km   irradiance {b['irradiance_uw_m2']:.0f} uW/m^2 "
          f"(floor in the standard: 25)   Prx {b['p_rx_dbm']:.1f} dBm")
    print(f"photons/bit {r['photons_per_bit']:.0f}   "
          f"pointing gain {r['pointing_gain_db'].mean():+.2f} dB mean, "
          f"{r['pointing_gain_db'].min():+.2f} dB worst frame")
    print(f"clock offset estimate {r['clock_ppm_est']:+.0f} ppm "
          f"(true {cfg.clock_ppm:+.0f})   headers {r['header_ok']}/{r['n_frames_counted']}")
    print(f"pre-FEC BER {r['pre_fec_ber']:.3e} "
          f"({r['pre_fec_errors']}/{r['pre_fec_bits']})")
    for name, d in r["post_fec"].items():
        print(f"post-FEC {name}: {d['errors']}/{d['bits']} bits, "
              f"{d['frame_errors']}/{d['frames']} frames bad")


if __name__ == "__main__":
    main()
