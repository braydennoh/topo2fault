"""run_all.py — full pipeline for the 15-river MHT topographic inversion.

Pipeline:
  1. River 11 (Budhi Gandaki) preflight — copy its original (147-pt)
     polyline back to the inversion input, re-snap to the COP30 DEM via
     D8 + 11x11 min-filter, then stage the snapped trace as the inversion
     input. The other 14 rivers already have their inversion inputs
     staged from the upstream pick.
  2. Run MCMC inversion for each of the 15 rivers (river_01..river_15),
     all with the same chain config: 30000 steps, 15000 burn, thin=30,
     480 walkers (~30 min/river → ~7.5 hr total). The full chain
     (thinned 50:1, including burn-in) is saved to mcmc_results.npz.

Run from the 5.17/ root: python3 run_all.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RIVERS_DIR = os.path.join(HERE, "data", "rivers")


def banner(s):
    print(f"\n{'='*70}\n {s}\n{'='*70}", flush=True)


def preflight_river_11():
    """River 11 (Budhi Gandaki) was extended upstream in an earlier
    experiment. Revert to the 147-pt original pick: copy orig→csv,
    re-snap to DEM, then snapped→csv (inversion input)."""
    banner("Preflight: river 11 (Budhi Gandaki) re-snap from original 147-pt pick")
    orig    = os.path.join(RIVERS_DIR, "river_11_orig.csv")
    csv     = os.path.join(RIVERS_DIR, "river_11.csv")
    snapped = os.path.join(RIVERS_DIR, "river_11_snapped.csv")

    shutil.copy2(orig, csv)
    from snap_rivers import snap_one_river
    t0 = time.time()
    r = snap_one_river(11)
    print(f"  snap: n={r['n_snap']}, len={r['len_snap_km']:.1f} km, "
          f"wall={time.time() - t0:.1f}s")
    shutil.copy2(snapped, csv)
    print(f"  staged: {csv} ({sum(1 for _ in open(csv)) - 1} rows)")


def run_inversion_for(rid: int):
    banner(f"Inversion: river_{rid:02d}")
    cmd = [sys.executable, "run_inversion.py",
           "--seg", str(rid), "--config", "F3"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=HERE, check=True)
    print(f"  wall: {(time.time() - t0) / 60.0:.1f} min", flush=True)


def main():
    t_all = time.time()
    preflight_river_11()
    for rid in range(1, 16):
        run_inversion_for(rid)
    banner(f"DONE — total wall {(time.time() - t_all) / 3600.0:.2f} hr")


if __name__ == "__main__":
    main()
