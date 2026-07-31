"""
snap_all_rivers.py — D8 channel-snap test for all 20 rivers.

For each river N:
  1. Crop COP30 DEM to a padded bbox of river_N.csv
  2. Fill sinks → D8 flow dir → flow accumulation → stream mask
  3. Snap the most-upstream and most-downstream river vertices to the
     stream network
  4. trace_downstream from the upstream snap, truncate at downstream snap
  5. Save the snapped channel polyline to
       river{N}/data/rivers/river_{NN}_snapped.csv
  6. Resample COP30 along original (densified) and snapped polylines with
     11x11 min-filter, plot both side-by-side and report HF residual stats
  7. Write per-river plot to riveraccuracy/river{NN}_compare.png

Final: one summary grid figure with all 20 panels at
  riveraccuracy/all_rivers_snap_compare.png
"""

from __future__ import annotations

import os
import sys
import gc
import time
import traceback
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds
from scipy.ndimage import minimum_filter, median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/braydennoh/Research/TheEntireHimalaya/3.6")
from topotoolbox_py import GRIDobj, FLOWobj, STREAMobj, fillsinks, flowacc

HERE = os.path.dirname(os.path.abspath(__file__))
DEM_COP = "/Users/braydennoh/Research/TheEntireHimalaya/tiffdata/COP30_MFT_full.tif"
OUT_DIR = os.path.join(HERE, "riveraccuracy")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD_CELLS = 30_000   # ≈ 27 km²; safe for all 20 rivers
PAD_DEG = 0.06             # bbox padding around each river polyline


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def crop_dem(src_path, dst_path, west, south, east, north):
    with rasterio.open(src_path) as src:
        win = from_bounds(west, south, east, north, transform=src.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        data = src.read(1, window=win).astype("int16")
        new_transform = src.window_transform(win)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": data.shape[0], "width": data.shape[1],
            "transform": new_transform, "compress": "lzw",
            "nodata": -32768,
        })
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(data, 1)


def snap_to_stream(mask, A, r, c, max_d=120):
    H, W = mask.shape
    for d in range(0, max_d):
        r0, r1 = max(0, r - d), min(H, r + d + 1)
        c0, c1 = max(0, c - d), min(W, c + d + 1)
        sub = mask[r0:r1, c0:c1]
        if sub.any():
            rr, cc = np.where(sub)
            accs = A[r0 + rr, c0 + cc]
            k = int(np.argmax(accs))
            return r0 + rr[k], c0 + cc[k]
    raise RuntimeError(f"No stream within {max_d} px of ({r},{c})")


def sample_cop_minfilter(lons, lats, k=11):
    with rasterio.open(DEM_COP) as src:
        H, W = src.shape
        rr, cc = rowcol(src.transform, lons, lats)
        rr = np.asarray(rr, int); cc = np.asarray(cc, int)
        r0, r1 = max(0, rr.min() - 30), min(H, rr.max() + 31)
        c0, c1 = max(0, cc.min() - 30), min(W, cc.max() + 31)
        dem = src.read(1, window=((r0, r1), (c0, c1))).astype(float)
        dem[dem < -100] = np.nan
    rl = rr - r0; cl = cc - c0
    mf = minimum_filter(dem, size=k, mode="nearest")
    return mf[rl, cl]


def hf_stats(elev):
    """High-frequency residual: deviation from a 200-pt median smooth."""
    if len(elev) < 50:
        return float("nan"), float("nan")
    sm = median_filter(elev, size=200, mode="nearest")
    res = elev - sm
    return float(np.nanstd(res)), float(np.nanpercentile(np.abs(res), 95))


# ───────────────────────────────────────────────────────────────────────
# Per-river snap
# ───────────────────────────────────────────────────────────────────────

def snap_one_river(rid: int):
    riv_csv = os.path.join(HERE, "data", "rivers", f"river_{rid:02d}.csv")
    if not os.path.exists(riv_csv):
        raise FileNotFoundError(riv_csv)
    riv = pd.read_csv(riv_csv)

    bbox = (riv.longitude.min() - PAD_DEG, riv.latitude.min() - PAD_DEG,
            riv.longitude.max() + PAD_DEG, riv.latitude.max() + PAD_DEG)
    crop_path = f"/tmp/cop30_river{rid:02d}_crop.tif"
    crop_dem(DEM_COP, crop_path, *bbox)

    DEM = GRIDobj(crop_path)
    DEMf = fillsinks(DEM)
    FD = FLOWobj(DEMf)
    A = flowacc(FD)
    mask = A > THRESHOLD_CELLS
    if not mask.any():
        # try a smaller threshold for tiny catchments
        mask = A > A.max() // 50
    S = STREAMobj(FD, mask)

    with rasterio.open(crop_path) as src:
        H, W = src.shape
        T = src.transform
    rows, cols = rowcol(T, riv.longitude.values, riv.latitude.values)
    rows = np.asarray(rows, int); cols = np.asarray(cols, int)

    if riv.elevation.values[0] > riv.elevation.values[-1]:
        up_idx, dn_idx = 0, len(riv) - 1
    else:
        up_idx, dn_idx = len(riv) - 1, 0
    r_up, c_up = snap_to_stream(mask, A, rows[up_idx], cols[up_idx])
    r_dn, c_dn = snap_to_stream(mask, A, rows[dn_idx], cols[dn_idx])
    flat_up = r_up * W + c_up
    flat_dn = r_dn * W + c_dn
    trace_idx = S.trace_downstream(flat_up)
    if flat_dn in trace_idx:
        cut = trace_idx.index(flat_dn) + 1
    else:
        tr_arr = np.asarray(trace_idx)
        tr_r = tr_arr // W; tr_c = tr_arr % W
        d2 = (tr_r - r_dn) ** 2 + (tr_c - c_dn) ** 2
        cut = int(np.argmin(d2)) + 1
    trace_idx = trace_idx[:cut]

    tr_arr = np.asarray(trace_idx)
    tr_r = tr_arr // W; tr_c = tr_arr % W
    xs, ys = rasterio.transform.xy(T, tr_r, tr_c)
    snap_lon = np.asarray(xs); snap_lat = np.asarray(ys)
    with rasterio.open(crop_path) as src:
        snap_elev = np.asarray(list(src.sample(zip(snap_lon, snap_lat))))[
            :, 0].astype(float)

    out_csv = riv_csv.replace(".csv", "_snapped.csv")
    pd.DataFrame({"longitude": snap_lon, "latitude": snap_lat,
                  "elevation": snap_elev}).to_csv(out_csv, index=False)

    # Build distance + COP30-min-filter samples for both polylines
    KM = 111.32 * np.cos(np.radians(snap_lat.mean()))
    s_orig = np.concatenate([[0],
        np.cumsum(np.hypot(np.diff(riv.longitude) * KM,
                            np.diff(riv.latitude) * 111.32))])
    n_dense = max(int(s_orig[-1] * 1000 / 30), 60)
    s_dense_o = np.linspace(0, s_orig[-1], n_dense)
    lons_o = np.interp(s_dense_o, s_orig, riv.longitude)
    lats_o = np.interp(s_dense_o, s_orig, riv.latitude)
    s_snap = np.concatenate([[0],
        np.cumsum(np.hypot(np.diff(snap_lon) * KM,
                            np.diff(snap_lat) * 111.32))])

    e_orig_11 = sample_cop_minfilter(lons_o, lats_o, k=11)
    e_snap_11 = sample_cop_minfilter(snap_lon, snap_lat, k=11)
    sd_o, p95_o = hf_stats(e_orig_11)
    sd_s, p95_s = hf_stats(e_snap_11)

    # cleanup big arrays
    del DEM, DEMf, FD, A, mask, S
    gc.collect()
    try:
        os.remove(crop_path)
    except OSError:
        pass

    return dict(
        rid=rid,
        len_orig_km=float(s_orig[-1]),
        len_snap_km=float(s_snap[-1]),
        n_orig=len(riv), n_snap=len(snap_lon),
        std_orig=sd_o, p95_orig=p95_o,
        std_snap=sd_s, p95_snap=p95_s,
        s_dense_o=s_dense_o, e_orig_11=e_orig_11,
        s_snap=s_snap, e_snap_11=e_snap_11,
    )


# ───────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────

def plot_one(result):
    rid = result["rid"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0),
                              sharex=False, sharey=True)
    axes[0].scatter(result["s_dense_o"], result["e_orig_11"],
                     s=1.2, c="steelblue", alpha=0.7)
    axes[0].set_title(
        f"river {rid:02d} — ORIGINAL polyline "
        f"({result['len_orig_km']:.0f} km, {result['n_orig']} CSV pts) "
        f"— HF resid std={result['std_orig']:.1f} m, p95={result['p95_orig']:.1f} m",
        loc="left", fontsize=8)
    axes[0].set_ylabel("Elev (m)")

    axes[1].scatter(result["s_snap"], result["e_snap_11"],
                     s=1.2, c="darkorange", alpha=0.7)
    axes[1].set_title(
        f"river {rid:02d} — SNAPPED polyline "
        f"({result['len_snap_km']:.0f} km, {result['n_snap']} pix) "
        f"— HF resid std={result['std_snap']:.1f} m, p95={result['p95_snap']:.1f} m",
        loc="left", fontsize=8)
    axes[1].set_xlabel("Distance from upstream (km)"); axes[1].set_ylabel("Elev (m)")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"river{rid:02d}_compare.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(results):
    """Single-page grid: 4 cols × 5 rows, each cell shows orig (blue) +
    snapped (orange) elevation traces."""
    fig, axes = plt.subplots(5, 4, figsize=(11, 10),
                              gridspec_kw=dict(hspace=0.55, wspace=0.20))
    for r in results:
        rid = r["rid"]
        row = (rid - 1) // 4
        col = (rid - 1) % 4
        ax = axes[row, col]
        ax.scatter(r["s_dense_o"], r["e_orig_11"], s=0.4,
                    c="steelblue", alpha=0.6, label="orig")
        ax.scatter(r["s_snap"], r["e_snap_11"], s=0.4,
                    c="darkorange", alpha=0.85, label="snap")
        ax.set_title(
            f"{rid:02d}  σ {r['std_orig']:.0f}→{r['std_snap']:.0f} m",
            fontsize=7, loc="left", pad=1)
        ax.tick_params(labelsize=5)
    for ax in axes.ravel():
        ax.set_xlabel("")
    handles = [
        plt.Line2D([0], [0], marker="o", lw=0, ms=3,
                   color="steelblue", label="original polyline"),
        plt.Line2D([0], [0], marker="o", lw=0, ms=3,
                   color="darkorange", label="D8-snapped polyline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.06)
    out_png = os.path.join(OUT_DIR, "all_rivers_snap_compare.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    results = []
    fails = []
    for rid in range(1, 21):
        t0 = time.time()
        try:
            r = snap_one_river(rid)
            plot_one(r)
            results.append(r)
            print(f"river{rid:02d}  orig: σ={r['std_orig']:5.1f} m / "
                  f"p95={r['p95_orig']:5.1f}    "
                  f"snap: σ={r['std_snap']:5.1f} m / "
                  f"p95={r['p95_snap']:5.1f}    "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            fails.append((rid, str(exc)))
            print(f"river{rid:02d}  FAILED  {exc}", flush=True)
            traceback.print_exc()
    if results:
        plot_grid(results)

    print()
    print(f"{'='*70}")
    print(f"{'riv':3} {'σ orig':>7} {'σ snap':>7} {'p95 orig':>9} {'p95 snap':>9}")
    for r in results:
        print(f"{r['rid']:3} {r['std_orig']:7.1f} {r['std_snap']:7.1f} "
              f"{r['p95_orig']:9.1f} {r['p95_snap']:9.1f}")
    if fails:
        print()
        print("FAILED:")
        for rid, msg in fails:
            print(f"  river{rid:02d}: {msg}")


if __name__ == "__main__":
    main()
