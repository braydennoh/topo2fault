"""
4.26/run_inversion.py — production real-river MCMC inversion.

Carries forward from `4.25/syn1/run_inversion.py`:
  - Real river ingest: lat/lon/elev → projected (sim_x, sim_y) + 3x3 min-filter
    DEM sampling.

Carries forward from `4.25/syn2/syn_inversion.py`:
  - **dip_min_deg = 3.0** (was 5.0; bug fix — strict > rejected exact-5°
    truth flats and inflated geometry by 1–3 km on the synthetic tests).
  - **`PosteriorStudentVarW_SigFixed_mnSampled`**: optional joint-(m, n)
    posterior, selectable by `--config F3`.
  - **Wider m prior**: `m ∈ [0.20, 1.20]` for F1/F2/F3 (was 0.20, 0.55).
  - **Saved derived-parameter posteriors** in `mcmc_results.npz`:
    `m_samples, n_samples, K_samples, ramp_x_samples, ramp_dip_samples`
    plus 16/50/84 percentiles.

This script ONLY runs the data ingest + MCMC + saves `mcmc_results.npz` and
`projected.csv` + `projected_check.pdf`. All the post-hoc figures
(`inversion.pdf`, `inversion_3panel.pdf`, `corner.pdf`, `corner_full.pdf`,
`posteriors.pdf`) are produced by the companion script `plot_inversion.py`.
That split lets you re-render figures without re-running the ~150 s MCMC.

CLI:
    python3 run_inversion.py --seg 1 --config F3
    python3 run_inversion.py --seg 1 --config F1 --quick
    python3 run_inversion.py --seg all --config F1
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.special import gammaln

import rasterio
from rasterio.windows import Window

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thrust_fault_model as tf  # noqa: E402
import emcee  # noqa: E402
import multiprocessing as mp  # noqa: E402

RIVERDATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
# DEM source: Copernicus GLO-30 (TanDEM-X-derived) — strict upgrade over SRTM GL1
# in steep Himalayan terrain. Tiles fetched from AWS Open Data and merged.
DEM_PATH = "/Users/braydennoh/Research/TheEntireHimalaya/tiffdata/COP30_MFT_full.tif"
OUT_BASE = RESULTS   # results land in results/river_{NN}/

# ── Model config ──
N_CTRL = 10
N_W = N_CTRL - 2
X_END_KM = 200.0
NU_FIXED = 5.0
SIGMA_FIXED = 1000.0
W_LO = 0.0
W_HI = 50.0           # 4.27: widened from 30
M_LO_F12 = 0.10
M_HI_F12 = 1.50
M_LO_F3 = 0.10        # 4.27: widened from 0.20
M_HI_F3 = 1.50        # 4.27: widened from 1.20
N_LO_F3 = 0.3         # 4.27: widened from 0.5
N_HI_F3 = 3.0         # 4.27: widened from 2.5

DIP_MIN_DEG = 1.0      # 4.27: widened from 3
DIP_MAX_DEG = 85.0     # 4.27: widened from 80

COMMON = dict(
    uplift_sample_axis="sim_y",
    structuralslip=1.0,
    hack_exp=1.7,
    l0_m=1_000.0,
    use_length_weights=True,
    eps=0.01,
    ld_lo=np.log(0.3),   # 4.27: widened from log(0.5)
    ld_hi=np.log(80.0),  # 4.27: widened from log(40)
)


# ───────────────────────────────────────────────────────────────────────
# Module-level forward-model helpers (must be picklable for fork pool).
# ───────────────────────────────────────────────────────────────────────

def build_c2_fault_model_varw(x, z, w_array):
    x = np.asarray(x, float); z = np.asarray(z, float)
    w_array = np.asarray(w_array, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    n = len(x)
    if n < 2: raise ValueError("Need at least 2 points")
    if len(w_array) != n - 2:
        raise ValueError(f"w_array length {len(w_array)} != n-2 = {n-2}")
    m_seg = np.diff(z) / np.diff(x)
    wv = np.zeros(n)
    for k, i in enumerate(range(1, n - 1)):
        wi = max(0.0, float(w_array[k]))
        wi = min(wi, 0.45 * (x[i] - x[i - 1]), 0.45 * (x[i + 1] - x[i]))
        wv[i] = wi
    patches = {}
    for i in range(1, n - 1):
        wi = wv[i]
        if wi <= 0: continue
        xL = x[i] - wi; xR = x[i] + wi
        mL = m_seg[i - 1]; mR = m_seg[i]
        zL = z[i] + mL * (xL - x[i])
        zR = z[i] + mR * (xR - x[i])
        coeffs, h = tf._quintic_patch_coeffs(xL, zL, mL, xR, zR, mR)
        patches[i] = (xL, coeffs, h)
    pieces: List[Tuple[str, float, float, int]] = []
    for seg in range(n - 1):
        left_cut = wv[seg] if seg > 0 else 0.0
        right_cut = wv[seg + 1] if (seg + 1) < n - 1 else 0.0
        xL = x[seg] + left_cut; xR = x[seg + 1] - right_cut
        if xR > xL:
            pieces.append(("line", xL, xR, seg))
        v = seg + 1
        if 1 <= v <= n - 2 and wv[v] > 0:
            pieces.append(("patch", x[v] - wv[v], x[v] + wv[v], v))
    return dict(x=x, z=z, m_seg=m_seg, pieces=pieces, patches=patches, wv=wv)


def compute_uplift_from_fault_varw(xctrl, zctrl, w_array, *,
                                    x_master, structuralslip,
                                    n_dense=600, n_nodes=350):
    x_master = np.asarray(x_master, float)
    m = build_c2_fault_model_varw(xctrl, zctrl, w_array)
    xd = np.linspace(float(np.min(xctrl)), float(np.max(xctrl)), int(n_dense))
    zd = tf.eval_c2_fault(xd, m)
    xn, zn, _, _ = tf.resample_equal_arclength(xd, zd, int(n_nodes))
    th, _, _ = tf.segment_angles(xn, zn)
    xa, _ = tf.compute_axial_surface_intersections(xn, zn, th)
    slip_by_seg = float(structuralslip) * np.ones_like(th)
    _, v, _ = tf.structural_velocity(
        x_master, th, xa, slip_by_seg, x_start=float(xn[0]))
    return v, (xn, zn)


def forward_model_onepass_varw(p_fault, w_array, *, data, fault_cfg,
                                x_master, structuralslip,
                                m_sp, n_sp, n_dense, n_nodes):
    try:
        xctrl, zctrl = tf.params_to_fault_ctrl(p_fault, cfg=fault_cfg)
    except Exception:
        return None
    if np.min(np.diff(xctrl)) < float(fault_cfg["dx_min_km"]): return None
    if not tf.dips_ok(xctrl, zctrl,
                      dip_min_deg=float(fault_cfg["dip_min_deg"]),
                      dip_max_deg=float(fault_cfg["dip_max_deg"])):
        return None
    if np.min(zctrl) < float(fault_cfg["zmin_ctrl_km"]): return None
    try:
        uplift_mm_yr, (xnode, znode) = compute_uplift_from_fault_varw(
            xctrl, zctrl, w_array,
            x_master=x_master, structuralslip=structuralslip,
            n_dense=n_dense, n_nodes=n_nodes)
    except Exception:
        return None
    if np.min(znode) < float(fault_cfg["zmin_node_km"]): return None
    uplift_on_channel = tf.interp_linear_extrap(
        data.x_uplift_coord_order, np.asarray(x_master, float), uplift_mm_yr)
    I = tf.stream_power_integral_I_from_uplift(
        uplift_on_channel, data.ds_m, data.A_mid, m_sp=m_sp, n_sp=n_sp)
    K_fit, z_pred, z0 = tf.fit_K_given_baselevel(I, data.Zobs_order, n_sp=n_sp)
    if not np.isfinite(K_fit): return None
    return dict(
        xctrl=xctrl, zctrl=zctrl,
        uplift_mm_yr=uplift_mm_yr, uplift_on_channel=uplift_on_channel,
        I=I, K_fit=float(K_fit), z_pred=z_pred, z0=float(z0),
        xnode=xnode, znode=znode)


MIN_FILTER_SIZE = 3         # DEM pixels (~30 m each); 3 → 90 m bed-snap window
# 4.27: SavGol outlier filter removed.  We now use the full dense densified
# topography directly with only the 3x3 min-filter applied per sample point.


def sample_min3x3(lons, lats, dem_path, size=MIN_FILTER_SIZE):
    """Sample DEM at each (lon, lat) using a min-filter window of `size`
    pixels on each side. The min-filter snaps the river point to the
    lowest pixel within the window — this corrects for HydroRIVERS
    centerline drift off the actual streambed in narrow gorges (where
    Copernicus DSM picks up canyon walls / canopy)."""
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    with rasterio.open(dem_path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows = np.asarray(rows, int); cols = np.asarray(cols, int)
        buf = max(3, size // 2 + 1)
        r0 = max(0, int(rows.min()) - buf)
        r1 = min(src.height, int(rows.max()) + buf + 1)
        c0 = max(0, int(cols.min()) - buf)
        c1 = min(src.width, int(cols.max()) + buf + 1)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        dem = src.read(1, window=win).astype(float)
    nrows, ncols = dem.shape
    dem[dem <= 0] = np.nan
    big = 1e18
    dem_filled = np.where(np.isnan(dem), big, dem)
    dem_min = minimum_filter(dem_filled, size=size, mode="nearest")
    dem_min = np.where(dem_min >= big, np.nan, dem_min)
    rows_local = np.clip(rows - r0, 0, nrows - 1)
    cols_local = np.clip(cols - c0, 0, ncols - 1)
    return dem[rows_local, cols_local], dem_min[rows_local, cols_local]


def ramp_center_and_dip(x, z, x_search_lo=30.0, x_search_hi=180.0):
    x = np.asarray(x); z = np.asarray(z)
    dx = np.diff(x); dz = np.diff(z)
    dip = np.degrees(np.arctan2(-dz, dx))
    xmid = 0.5 * (x[:-1] + x[1:])
    mask = (xmid >= x_search_lo) & (xmid <= x_search_hi)
    if not np.any(mask):
        return float("nan"), float("nan")
    sub_x = xmid[mask]; sub_d = dip[mask]
    peak = float(np.max(sub_d))
    floor_ = float(np.min(sub_d))
    thr = 0.5 * (floor_ + peak)
    above = np.where(sub_d >= thr)[0]
    if above.size == 0:
        return float(sub_x[int(np.argmax(sub_d))]), peak
    return 0.5 * (float(sub_x[above[0]]) + float(sub_x[above[-1]])), peak


# ───────────────────────────────────────────────────────────────────────
# Posterior classes
# ───────────────────────────────────────────────────────────────────────

class PosteriorStudentVarW_SigFixed_mSampled:
    """θ = [q (nq), ld (nld), m, w (nw)]   (n is fixed)."""

    def __init__(self, *, data, fault_cfg, x_master, structuralslip,
                 eps, ld_lo, ld_hi, n_fixed, nu_fixed, sigma_fixed,
                 m_lo, m_hi, w_lo, w_hi, n_w,
                 n_dense_like=600, n_nodes_like=350):
        self.data = data; self.fault_cfg = fault_cfg
        self.x_master = x_master; self.structuralslip = structuralslip
        self.eps = eps; self.ld_lo = ld_lo; self.ld_hi = ld_hi
        self.n_fixed = float(n_fixed); self.nu_fixed = float(nu_fixed)
        self.sigma_fixed = float(sigma_fixed)
        self.m_lo = float(m_lo); self.m_hi = float(m_hi)
        self.w_lo = float(w_lo); self.w_hi = float(w_hi); self.n_w = int(n_w)
        self.n_dense_like = n_dense_like; self.n_nodes_like = n_nodes_like
        n_q, n_ld = tf.fault_param_counts(fault_cfg)
        self.n_q = int(n_q); self.n_ld = int(n_ld)
        self.n_fault_params = int(n_q + n_ld)
        self.ndim = self.n_fault_params + 1 + self.n_w
        nu = self.nu_fixed
        self._t_const = (gammaln((nu+1)/2.0) - gammaln(nu/2.0)
                         - 0.5*np.log(np.pi*nu) - np.log(self.sigma_fixed))

    def _split(self, theta):
        theta = np.asarray(theta, float); i0 = self.n_fault_params
        return theta[:i0], float(theta[i0]), theta[i0+1:i0+1+self.n_w]

    def log_prior(self, theta):
        p_fault, m_r, w_array = self._split(theta)
        q = p_fault[:self.n_q]; ld = p_fault[self.n_q:]
        if np.any(q <= self.eps) or np.any(q >= 1.0 - self.eps): return -np.inf
        if np.any(ld <= self.ld_lo) or np.any(ld >= self.ld_hi): return -np.inf
        if not (self.m_lo <= m_r <= self.m_hi): return -np.inf
        if np.any(w_array < self.w_lo) or np.any(w_array > self.w_hi): return -np.inf
        try:
            x, z = tf.params_to_fault_ctrl(p_fault, cfg=self.fault_cfg)
        except Exception:
            return -np.inf
        if np.min(np.diff(x)) < float(self.fault_cfg["dx_min_km"]): return -np.inf
        if not tf.dips_ok(x, z,
                          dip_min_deg=float(self.fault_cfg["dip_min_deg"]),
                          dip_max_deg=float(self.fault_cfg["dip_max_deg"])):
            return -np.inf
        if np.min(z) < float(self.fault_cfg["zmin_ctrl_km"]): return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        p_fault, m_r, w_array = self._split(theta)
        out = forward_model_onepass_varw(
            p_fault, w_array, data=self.data, fault_cfg=self.fault_cfg,
            x_master=self.x_master, structuralslip=self.structuralslip,
            m_sp=m_r, n_sp=self.n_fixed,
            n_dense=self.n_dense_like, n_nodes=self.n_nodes_like)
        if out is None: return -np.inf
        r = self.data.Zobs_order - out["z_pred"]
        sigma = self.sigma_fixed; nu = self.nu_fixed
        logpdf = self._t_const - 0.5*(nu+1.0)*np.log1p((r/sigma)**2 / nu)
        return float(np.sum(self.data.w_node * logpdf))

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll): return -np.inf
        return float(lp + ll)

    def draw_initial_ball(self, rng, n_walkers, *,
                          planar_dip_deg=7.0,
                          spread_q=0.015, spread_ld=0.12,
                          spread_m=0.03, spread_w=3.0,
                          max_tries=500):
        n = int(self.fault_cfg["n_ctrl"]); x_end = float(self.fault_cfg["x_end_km"])
        xctrl_ref = np.linspace(0.0, x_end, n)
        q_ref = np.zeros(self.n_q)
        free_idx = tf.fault_free_x_indices(self.fault_cfg)
        for idx, i in enumerate(free_idx):
            q_ref[idx] = ((xctrl_ref[i] - xctrl_ref[i-1]) / (x_end - xctrl_ref[i-1]))
        dx_seg = x_end / (n - 1)
        dz_seg = dx_seg * np.tan(np.radians(planar_dip_deg))
        if dz_seg <= 0: raise ValueError("planar_dip_deg must be > 0")
        ld_ref = np.log(dz_seg) * np.ones(self.n_ld)
        m_ref = 0.5 * (self.m_lo + self.m_hi)
        w_ref_arr = 0.5 * (self.w_lo + self.w_hi) * np.ones(self.n_w)
        theta_ref = np.concatenate([q_ref, ld_ref, [m_ref], w_ref_arr])
        if not np.isfinite(self(theta_ref)):
            raise RuntimeError(
                f"Reference theta (planar dip {planar_dip_deg}°) invalid.")
        walkers = np.zeros((n_walkers, self.ndim))
        fails = 0
        for w in range(n_walkers):
            for _ in range(max_tries):
                noise = np.concatenate([
                    rng.normal(0, spread_q, self.n_q),
                    rng.normal(0, spread_ld, self.n_ld),
                    [rng.normal(0, spread_m)],
                    rng.normal(0, spread_w, self.n_w),
                ])
                theta = theta_ref + noise
                if np.isfinite(self(theta)):
                    walkers[w] = theta; break
            else:
                fails += 1; walkers[w] = theta_ref.copy()
        if fails:
            print(f"  WARN: {fails}/{n_walkers} walkers fell back to reference",
                  flush=True)
        return walkers

    def evaluate(self, theta, *, n_dense=600, n_nodes=350):
        p_fault, m_r, w_array = self._split(theta)
        return forward_model_onepass_varw(
            p_fault, w_array, data=self.data, fault_cfg=self.fault_cfg,
            x_master=self.x_master, structuralslip=self.structuralslip,
            m_sp=m_r, n_sp=self.n_fixed,
            n_dense=n_dense, n_nodes=n_nodes)


class PosteriorStudentVarW_SigFixed_mnSampled:
    """θ = [q (nq), ld (nld), m, n, w (nw)]   (m and n both sampled)."""

    def __init__(self, *, data, fault_cfg, x_master, structuralslip,
                 eps, ld_lo, ld_hi, n_lo, n_hi, nu_fixed, sigma_fixed,
                 m_lo, m_hi, w_lo, w_hi, n_w,
                 n_dense_like=600, n_nodes_like=350):
        self.data = data; self.fault_cfg = fault_cfg
        self.x_master = x_master; self.structuralslip = structuralslip
        self.eps = eps; self.ld_lo = ld_lo; self.ld_hi = ld_hi
        self.n_lo = float(n_lo); self.n_hi = float(n_hi)
        self.nu_fixed = float(nu_fixed); self.sigma_fixed = float(sigma_fixed)
        self.m_lo = float(m_lo); self.m_hi = float(m_hi)
        self.w_lo = float(w_lo); self.w_hi = float(w_hi); self.n_w = int(n_w)
        self.n_dense_like = n_dense_like; self.n_nodes_like = n_nodes_like
        n_q, n_ld = tf.fault_param_counts(fault_cfg)
        self.n_q = int(n_q); self.n_ld = int(n_ld)
        self.n_fault_params = int(n_q + n_ld)
        self.ndim = self.n_fault_params + 2 + self.n_w
        nu = self.nu_fixed
        self._t_const = (gammaln((nu+1)/2.0) - gammaln(nu/2.0)
                         - 0.5*np.log(np.pi*nu) - np.log(self.sigma_fixed))

    def _split(self, theta):
        theta = np.asarray(theta, float); i0 = self.n_fault_params
        return (theta[:i0], float(theta[i0]), float(theta[i0+1]),
                theta[i0+2:i0+2+self.n_w])

    def log_prior(self, theta):
        p_fault, m_r, n_r, w_array = self._split(theta)
        q = p_fault[:self.n_q]; ld = p_fault[self.n_q:]
        if np.any(q <= self.eps) or np.any(q >= 1.0 - self.eps): return -np.inf
        if np.any(ld <= self.ld_lo) or np.any(ld >= self.ld_hi): return -np.inf
        if not (self.m_lo <= m_r <= self.m_hi): return -np.inf
        if not (self.n_lo <= n_r <= self.n_hi): return -np.inf
        if np.any(w_array < self.w_lo) or np.any(w_array > self.w_hi): return -np.inf
        try:
            x, z = tf.params_to_fault_ctrl(p_fault, cfg=self.fault_cfg)
        except Exception:
            return -np.inf
        if np.min(np.diff(x)) < float(self.fault_cfg["dx_min_km"]): return -np.inf
        if not tf.dips_ok(x, z,
                          dip_min_deg=float(self.fault_cfg["dip_min_deg"]),
                          dip_max_deg=float(self.fault_cfg["dip_max_deg"])):
            return -np.inf
        if np.min(z) < float(self.fault_cfg["zmin_ctrl_km"]): return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        p_fault, m_r, n_r, w_array = self._split(theta)
        out = forward_model_onepass_varw(
            p_fault, w_array, data=self.data, fault_cfg=self.fault_cfg,
            x_master=self.x_master, structuralslip=self.structuralslip,
            m_sp=m_r, n_sp=n_r,
            n_dense=self.n_dense_like, n_nodes=self.n_nodes_like)
        if out is None: return -np.inf
        r = self.data.Zobs_order - out["z_pred"]
        sigma = self.sigma_fixed; nu = self.nu_fixed
        logpdf = self._t_const - 0.5*(nu+1.0)*np.log1p((r/sigma)**2 / nu)
        return float(np.sum(self.data.w_node * logpdf))

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll): return -np.inf
        return float(lp + ll)

    def draw_initial_ball(self, rng, n_walkers, *,
                          planar_dip_deg=7.0,
                          spread_q=0.015, spread_ld=0.12,
                          spread_m=0.03, spread_n=0.10, spread_w=3.0,
                          max_tries=500):
        n = int(self.fault_cfg["n_ctrl"]); x_end = float(self.fault_cfg["x_end_km"])
        xctrl_ref = np.linspace(0.0, x_end, n)
        q_ref = np.zeros(self.n_q)
        free_idx = tf.fault_free_x_indices(self.fault_cfg)
        for idx, i in enumerate(free_idx):
            q_ref[idx] = ((xctrl_ref[i] - xctrl_ref[i-1]) / (x_end - xctrl_ref[i-1]))
        dx_seg = x_end / (n - 1)
        dz_seg = dx_seg * np.tan(np.radians(planar_dip_deg))
        ld_ref = np.log(dz_seg) * np.ones(self.n_ld)
        m_ref = 0.5 * (self.m_lo + self.m_hi)
        n_ref = 0.5 * (self.n_lo + self.n_hi)
        w_ref_arr = 0.5 * (self.w_lo + self.w_hi) * np.ones(self.n_w)
        theta_ref = np.concatenate([q_ref, ld_ref, [m_ref, n_ref], w_ref_arr])
        if not np.isfinite(self(theta_ref)):
            raise RuntimeError(
                f"Reference theta (planar dip {planar_dip_deg}°) invalid.")
        walkers = np.zeros((n_walkers, self.ndim))
        fails = 0
        for w in range(n_walkers):
            for _ in range(max_tries):
                noise = np.concatenate([
                    rng.normal(0, spread_q, self.n_q),
                    rng.normal(0, spread_ld, self.n_ld),
                    [rng.normal(0, spread_m), rng.normal(0, spread_n)],
                    rng.normal(0, spread_w, self.n_w),
                ])
                theta = theta_ref + noise
                if np.isfinite(self(theta)):
                    walkers[w] = theta; break
            else:
                fails += 1; walkers[w] = theta_ref.copy()
        if fails:
            print(f"  WARN: {fails}/{n_walkers} walkers fell back to reference",
                  flush=True)
        return walkers

    def evaluate(self, theta, *, n_dense=600, n_nodes=350):
        p_fault, m_r, n_r, w_array = self._split(theta)
        return forward_model_onepass_varw(
            p_fault, w_array, data=self.data, fault_cfg=self.fault_cfg,
            x_master=self.x_master, structuralslip=self.structuralslip,
            m_sp=m_r, n_sp=n_r,
            n_dense=n_dense, n_nodes=n_nodes)


# ───────────────────────────────────────────────────────────────────────
# Real-data ingest
# ───────────────────────────────────────────────────────────────────────

DENSIFY_SPACING_M = 30.0   # match DEM resolution


def densify_polyline(lons, lats, target_spacing_m, lat_mean=29.0):
    """Resample a (lon, lat) polyline at uniform along-track spacing.
    Linear interpolation between original vertices — preserves the
    centerline geometry while increasing point density for DEM sampling.
    """
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    deg2km_lat = 111.32
    deg2km_lon = 111.32 * np.cos(np.radians(lat_mean))
    dlons = np.diff(lons) * deg2km_lon * 1000.0
    dlats = np.diff(lats) * deg2km_lat * 1000.0
    seg_lens = np.sqrt(dlons**2 + dlats**2)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum_s[-1])
    if total <= 0:
        return lons.copy(), lats.copy()
    n_new = max(int(np.ceil(total / float(target_spacing_m))) + 1, 2)
    s_new = np.linspace(0.0, total, n_new)
    lons_new = np.interp(s_new, cum_s, lons)
    lats_new = np.interp(s_new, cum_s, lats)
    return lons_new, lats_new


def project_river_to_fault_axis(seg_id: int, out_dir: str):
    """Read normals.csv + river_<seg>.csv, densify the polyline at
    `DENSIFY_SPACING_M` to match DEM resolution, project to
    (sim_x, sim_y), DEM-sample with 3x3 minimum filter, write
    projected.csv + projected_check.pdf."""

    normals = pd.read_csv(os.path.join(RIVERDATA, "normals.csv"))
    seg = normals[normals.normal_id == seg_id].iloc[0]
    fault_lon, fault_lat = seg.base_lon, seg.base_lat
    tip_lon, tip_lat = seg.tip_lon, seg.tip_lat

    lat_mean = 29.0
    deg2km_lat = 111.32
    deg2km_lon = 111.32 * np.cos(np.radians(lat_mean))

    n_east = (tip_lon - fault_lon) * deg2km_lon
    n_north = (tip_lat - fault_lat) * deg2km_lat
    n_mag = np.sqrt(n_east**2 + n_north**2)
    n_hat = np.array([n_east, n_north]) / n_mag
    s_hat = np.array([-n_hat[1], n_hat[0]])

    river = pd.read_csv(os.path.join(RIVERDATA, "rivers", f"river_{seg_id:02d}.csv"))
    lons_raw = river.longitude.values.astype(float)
    lats_raw = river.latitude.values.astype(float)
    orig_elev_raw = river.elevation.values.astype(float)

    # Densify along-track to match DEM resolution
    lons, lats = densify_polyline(lons_raw, lats_raw, DENSIFY_SPACING_M,
                                   lat_mean=lat_mean)
    print(f"[Seg {seg_id}] {len(river)} CSV pts → {len(lons)} densified pts "
          f"(@ {DENSIFY_SPACING_M:.0f} m), sampling DEM with "
          f"{MIN_FILTER_SIZE}x{MIN_FILTER_SIZE} min-filter "
          f"(~{MIN_FILTER_SIZE * 30} m window)…",
          flush=True)
    # Carry the original-CSV elevations only as a diagnostic by mapping each
    # densified point to the nearest original vertex (for the bottom panels).
    seg_lens_raw = np.sqrt(
        (np.diff(lons_raw) * deg2km_lon * 1000.0) ** 2 +
        (np.diff(lats_raw) * deg2km_lat * 1000.0) ** 2)
    cum_s_raw = np.concatenate([[0.0], np.cumsum(seg_lens_raw)])
    seg_lens_new = np.sqrt(
        (np.diff(lons) * deg2km_lon * 1000.0) ** 2 +
        (np.diff(lats) * deg2km_lat * 1000.0) ** 2)
    cum_s_new = np.concatenate([[0.0], np.cumsum(seg_lens_new)])
    orig_elev = np.interp(cum_s_new, cum_s_raw, orig_elev_raw)

    elev_raw_full, elev_min_full = sample_min3x3(lons, lats, DEM_PATH)
    lons_full = lons.copy(); lats_full = lats.copy()

    # 4.27: no SavGol outlier filter — fit the full dense profile.  Only
    # filtering applied is the 3x3 min-filter at sample time.
    n_total = len(elev_min_full)
    elev_raw = elev_raw_full
    elev_used = elev_min_full          # ORIGINAL DEM value (no sparsification)
    print(f"  using full dense profile: {n_total} pts (no SavGol)",
          flush=True)

    diff = orig_elev - elev_used
    print(f"  orig − used: mean={np.nanmean(diff):.1f} m, "
          f"median={np.nanmedian(diff):.1f} m, "
          f"p95={np.nanpercentile(diff, 95):.1f} m", flush=True)

    dx_km = (lons - fault_lon) * deg2km_lon
    dy_km = (lats - fault_lat) * deg2km_lat
    sim_y = dx_km * n_hat[0] + dy_km * n_hat[1]
    sim_x = dx_km * s_hat[0] + dy_km * s_hat[1]

    proj_path = os.path.join(out_dir, "projected.csv")
    pd.DataFrame({
        "sim_x": sim_x, "sim_y": sim_y,
        "elevation": elev_used,
        "elev_raw_nn": elev_raw,
        "elev_orig_csv": orig_elev,
    }).to_csv(proj_path, index=False)

    s_arc = tf.cumulative_arclength(sim_x, sim_y)
    outlet_is_start = bool(elev_used[0] <= elev_used[-1])
    s_from_outlet = s_arc if outlet_is_start else (s_arc[-1] - s_arc)
    order = np.argsort(s_from_outlet)

    # 4.27 diagnostic: just show the full dense profile that's being fit.
    figP, axesP = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axesP[0]
    ax.plot(lons_full, lats_full, '-', color='0.6', lw=0.5,
            label='dense polyline')
    ax.plot([fault_lon, tip_lon], [fault_lat, tip_lat], 'b-', lw=2,
            label='normal')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_aspect(1.0 / np.cos(np.radians(lat_mean)))
    ax.legend(fontsize=7)

    ax = axesP[1]
    sc = ax.scatter(sim_x, sim_y, c=elev_used, s=1, cmap='viridis')
    ax.set_xlabel('sim_x (km)'); ax.set_ylabel('sim_y (km)')
    plt.colorbar(sc, ax=ax, label='Elev (m)')

    ax = axesP[2]
    ax.scatter(s_from_outlet[order], elev_used[order],
               s=1, c='steelblue',
               label=f'{MIN_FILTER_SIZE}x{MIN_FILTER_SIZE} min ({n_total} pts)')
    ax.set_xlabel('Distance from outlet (km)'); ax.set_ylabel('Elev (m)')
    ax.legend(fontsize=7)

    figP.tight_layout()
    figP.savefig(os.path.join(out_dir, "projected_check.pdf"),
                 bbox_inches='tight')
    plt.close(figP)
    return proj_path


# ───────────────────────────────────────────────────────────────────────
# fault_cfg + posterior factory
# ───────────────────────────────────────────────────────────────────────

def fault_cfg_default():
    return dict(
        x_end_km=float(X_END_KM),
        n_ctrl=N_CTRL,
        x_fixed={0: 0.0, -1: float(X_END_KM)},
        z_fixed={0: 0.0},
        dx_min_km=10.0,
        dip_min_deg=DIP_MIN_DEG,
        dip_max_deg=DIP_MAX_DEG,
        zmin_ctrl_km=-30.0,
        zmin_node_km=-40.0,
        z_end_soft_km=-50.0,
        z_end_soft_wt=0.02,
    )


def make_posterior(config: str, *, data, x_master, structuralslip):
    cfg = fault_cfg_default()
    if config == "F1":
        return cfg, PosteriorStudentVarW_SigFixed_mSampled(
            data=data, fault_cfg=cfg, x_master=x_master,
            structuralslip=structuralslip,
            eps=COMMON["eps"], ld_lo=COMMON["ld_lo"], ld_hi=COMMON["ld_hi"],
            n_fixed=1.0, nu_fixed=NU_FIXED, sigma_fixed=SIGMA_FIXED,
            m_lo=M_LO_F12, m_hi=M_HI_F12,
            w_lo=W_LO, w_hi=W_HI, n_w=N_W)
    if config == "F2":
        return cfg, PosteriorStudentVarW_SigFixed_mSampled(
            data=data, fault_cfg=cfg, x_master=x_master,
            structuralslip=structuralslip,
            eps=COMMON["eps"], ld_lo=COMMON["ld_lo"], ld_hi=COMMON["ld_hi"],
            n_fixed=2.0, nu_fixed=NU_FIXED, sigma_fixed=SIGMA_FIXED,
            m_lo=M_LO_F12, m_hi=M_HI_F12,
            w_lo=W_LO, w_hi=W_HI, n_w=N_W)
    if config == "F3":
        return cfg, PosteriorStudentVarW_SigFixed_mnSampled(
            data=data, fault_cfg=cfg, x_master=x_master,
            structuralslip=structuralslip,
            eps=COMMON["eps"], ld_lo=COMMON["ld_lo"], ld_hi=COMMON["ld_hi"],
            n_lo=N_LO_F3, n_hi=N_HI_F3,
            nu_fixed=NU_FIXED, sigma_fixed=SIGMA_FIXED,
            m_lo=M_LO_F3, m_hi=M_HI_F3,
            w_lo=W_LO, w_hi=W_HI, n_w=N_W)
    raise ValueError(f"unknown config: {config!r}")


def median_ctrl_points(flat_s, fault_cfg, *, stride=10):
    n_q, n_ld = tf.fault_param_counts(fault_cfg)
    n_fault = n_q + n_ld
    xs, zs = [], []
    for s in flat_s[::stride]:
        try:
            x, z = tf.params_to_fault_ctrl(s[:n_fault], cfg=fault_cfg)
        except Exception:
            continue
        xs.append(x); zs.append(z)
    return np.median(np.asarray(xs), axis=0), np.median(np.asarray(zs), axis=0)


# ───────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────

def run_one(seg_id: int, config: str, *, quick: bool = False, seed: int = 42):
    out_dir = os.path.join(OUT_BASE, f"river_{seg_id:02d}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== run_one(seg={seg_id}, config={config}, quick={quick}) ===",
          flush=True)
    print(f"  output dir: {out_dir}", flush=True)
    print(f"  fault_cfg dip_min_deg = {DIP_MIN_DEG}", flush=True)

    t0 = time.time()
    proj_path = project_river_to_fault_axis(seg_id, out_dir)
    data = tf.load_channel_profile_csv(
        proj_path,
        uplift_sample_axis=COMMON["uplift_sample_axis"],
        use_length_weights=COMMON["use_length_weights"],
        hack_exp=COMMON["hack_exp"], l0_m=COMMON["l0_m"],
    )
    print(f"  data: {len(data.X)} pts, stream len {data.s_km_model[-1]:.1f} km, "
          f"outlet_is_start={data.outlet_is_start}", flush=True)

    x_master = np.linspace(1e-1, X_END_KM, 1000)
    fault_cfg, posterior = make_posterior(
        config, data=data, x_master=x_master,
        structuralslip=COMMON["structuralslip"])
    print(f"  posterior: {type(posterior).__name__}, ndim={posterior.ndim}",
          flush=True)

    if quick:
        nsteps, burn, thin = 1500, 1000, 5
        nwalkers = max(40, 2 * posterior.ndim + 2)
    else:
        # 5.17: 3x current full-mode chain length for the convergence
        # animation. 480 walkers × 30000 steps ≈ 14M evals (~30 min/river).
        # Burn 15000 leaves 15000 post-burn per walker → 15000/30 thin
        # × 480 = 240k samples (same density as before, longer chain).
        nsteps, burn, thin = 30000, 15000, 30
        nwalkers = max(480, 2 * posterior.ndim + 2)
    # Animation snapshot stride — save the full chain (including burn-in)
    # at this stride so we can replay walker positions vs MCMC step.
    chain_anim_thin = 50
    print(f"  MCMC: nsteps={nsteps}, burn={burn}, thin={thin}, "
          f"nwalkers={nwalkers}, anim_thin={chain_anim_thin}", flush=True)

    rng = np.random.default_rng(seed)
    p0 = posterior.draw_initial_ball(rng, nwalkers, planar_dip_deg=7.0)

    nproc = max(1, mp.cpu_count() - 1)
    print(f"  pool: {nproc} processes (fork)", flush=True)
    ctx = mp.get_context("fork") if sys.platform == "darwin" else mp.get_context()
    t_mcmc = time.time()
    with ctx.Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, posterior.ndim, posterior, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=False)
    print(f"  mcmc: {time.time() - t_mcmc:.1f} s", flush=True)

    flat_s = sampler.get_chain(discard=burn, thin=thin, flat=True)
    flat_lp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)

    # Full chain history (including burn-in) for the convergence animation.
    # Shape: (nsteps/chain_anim_thin, nwalkers, ndim).
    chain_anim = sampler.get_chain(thin=chain_anim_thin, flat=False)
    chain_anim_lp = sampler.get_log_prob(thin=chain_anim_thin, flat=False)

    # Best (MAP) sample with valid forward
    order_lp = np.argsort(flat_lp)[::-1]
    best, th_best = None, None
    for idx in order_lp[:500]:
        th_best = flat_s[idx]
        best = posterior.evaluate(th_best, n_dense=600, n_nodes=350)
        if best is not None: break
    if best is None:
        print("  FAILED — no valid MAP sample", flush=True)
        return None

    # Posterior subsample evaluations
    rng2 = np.random.default_rng(0)
    nsample = min(300, len(flat_s))
    post_idx = rng2.choice(len(flat_s), size=nsample, replace=True)

    is_F3 = isinstance(posterior, PosteriorStudentVarW_SigFixed_mnSampled)
    z_pred_all, uplift_all, xnode_all, znode_all = [], [], [], []
    K_all, ramp_x_all, ramp_dip_all = [], [], []
    m_all, n_all = [], []
    for th in flat_s[post_idx]:
        out = posterior.evaluate(th, n_dense=600, n_nodes=350)
        if out is None: continue
        z_pred_all.append(out["z_pred"])
        uplift_all.append(out["uplift_mm_yr"])
        xnode_all.append(out["xnode"]); znode_all.append(out["znode"])
        K_all.append(out["K_fit"])
        cx, dd = ramp_center_and_dip(out["xnode"], out["znode"])
        ramp_x_all.append(cx); ramp_dip_all.append(dd)
        if is_F3:
            _, m_th, n_th, _ = posterior._split(th)
        else:
            _, m_th, _ = posterior._split(th); n_th = posterior.n_fixed
        m_all.append(m_th); n_all.append(n_th)

    z_pred_all = np.asarray(z_pred_all)
    uplift_all = np.asarray(uplift_all)
    K_all = np.asarray(K_all)
    ramp_x_all = np.asarray(ramp_x_all); ramp_dip_all = np.asarray(ramp_dip_all)
    m_all = np.asarray(m_all); n_all = np.asarray(n_all)

    z_pred_median = np.median(z_pred_all, axis=0)
    z_pred_lo = np.percentile(z_pred_all, 16, axis=0)
    z_pred_hi = np.percentile(z_pred_all, 84, axis=0)
    uplift_median = np.median(uplift_all, axis=0)
    uplift_lo = np.percentile(uplift_all, 16, axis=0)
    uplift_hi = np.percentile(uplift_all, 84, axis=0)

    xnode_common = np.linspace(0.0, X_END_KM, 500)
    znode_interp = np.array([np.interp(xnode_common, xn, zn)
                             for xn, zn in zip(xnode_all, znode_all)])
    znode_median = np.median(znode_interp, axis=0)
    znode_lo = np.percentile(znode_interp, 16, axis=0)
    znode_hi = np.percentile(znode_interp, 84, axis=0)

    rmse_best = float(np.sqrt(np.mean((data.Zobs_order - best["z_pred"])**2)))
    rmse_median = float(np.sqrt(np.mean((data.Zobs_order - z_pred_median)**2)))

    m_q = np.percentile(m_all, [16, 50, 84])
    n_q = np.percentile(n_all, [16, 50, 84])
    K_q = np.percentile(K_all, [16, 50, 84])
    ramp_x_q = np.percentile(ramp_x_all, [16, 50, 84])
    ramp_dip_q = np.percentile(ramp_dip_all, [16, 50, 84])

    print(f"  rmse_profile (m): best={rmse_best:.1f}, median={rmse_median:.1f}",
          flush=True)
    print(f"  m   16/50/84 = {m_q[0]:.3f}/{m_q[1]:.3f}/{m_q[2]:.3f}", flush=True)
    if is_F3:
        print(f"  n   16/50/84 = {n_q[0]:.3f}/{n_q[1]:.3f}/{n_q[2]:.3f}", flush=True)
    print(f"  K   16/50/84 = {K_q[0]:.2e}/{K_q[1]:.2e}/{K_q[2]:.2e}", flush=True)
    print(f"  ramp_x 16/50/84 = "
          f"{ramp_x_q[0]:.1f}/{ramp_x_q[1]:.1f}/{ramp_x_q[2]:.1f} km", flush=True)
    print(f"  ramp_dip 16/50/84 = "
          f"{ramp_dip_q[0]:.2f}/{ramp_dip_q[1]:.2f}/{ramp_dip_q[2]:.2f} °", flush=True)

    median_xctrl, median_zctrl = median_ctrl_points(flat_s, fault_cfg)

    np.savez(
        os.path.join(out_dir, "mcmc_results.npz"),
        seg_id=seg_id, config=config, ndim=posterior.ndim,
        is_F3=is_F3,
        n_q_dim=posterior.n_q, n_ld_dim=posterior.n_ld, n_w_dim=posterior.n_w,
        flat_s=flat_s, flat_lp=flat_lp, th_best=th_best,
        m_samples=m_all, n_samples=n_all, K_samples=K_all,
        ramp_x_samples=ramp_x_all, ramp_dip_samples=ramp_dip_all,
        m_q=m_q, n_q=n_q, K_q=K_q,
        ramp_x_q=ramp_x_q, ramp_dip_q=ramp_dip_q,
        x_master=x_master,
        s_km_model=data.s_km_model,
        Zobs_order=data.Zobs_order,
        x_uplift_coord_order=data.x_uplift_coord_order,
        z_pred_best=best["z_pred"], z_pred_median=z_pred_median,
        z_pred_lo=z_pred_lo, z_pred_hi=z_pred_hi,
        uplift_best=best["uplift_mm_yr"], uplift_median=uplift_median,
        uplift_lo=uplift_lo, uplift_hi=uplift_hi,
        xctrl_best=best["xctrl"], zctrl_best=best["zctrl"],
        xnode_best=best["xnode"], znode_best=best["znode"],
        xnode_common=xnode_common, znode_median=znode_median,
        znode_lo=znode_lo, znode_hi=znode_hi,
        median_xctrl=median_xctrl, median_zctrl=median_zctrl,
        z_pred_samples=z_pred_all, uplift_samples=uplift_all,
        xnode_samples=np.asarray(xnode_all),
        znode_samples=np.asarray(znode_all),
        rmse_best=rmse_best, rmse_median=rmse_median,
        nsteps=nsteps, burn=burn, thin=thin, nwalkers=nwalkers,
        chain_anim=chain_anim, chain_anim_lp=chain_anim_lp,
        chain_anim_thin=chain_anim_thin,
        dip_min_deg=DIP_MIN_DEG,
        x_end_km=X_END_KM,
    )

    print(f"  wall: {time.time() - t0:.1f} s. Saved {out_dir}/mcmc_results.npz",
          flush=True)
    print(f"  → run `python3 plot_inversion.py --dir {out_dir}` to render figures",
          flush=True)
    return dict(out_dir=out_dir, rmse_best=rmse_best, rmse_median=rmse_median,
                wall_s=time.time() - t0)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--seg", type=str, default="1",
                   help="segment id (e.g. '1') or 'all'")
    p.add_argument("--config", type=str, default="F3",
                   choices=["F1", "F2", "F3", "all"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true",
                   help="auto-run plot_inversion.py on each output dir")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.seg == "all":
        rivers_dir = os.path.join(RIVERDATA, "rivers")
        seg_ids = sorted(int(f[6:8]) for f in os.listdir(rivers_dir)
                         if f.startswith("river_") and f.endswith(".csv"))
    else:
        seg_ids = [int(args.seg)]
    configs = ["F1", "F2", "F3"] if args.config == "all" else [args.config]
    summary = []
    for sid in seg_ids:
        for cfg in configs:
            res = run_one(sid, cfg, quick=args.quick, seed=args.seed)
            summary.append((sid, cfg, res))
            if args.plot and res is not None:
                import subprocess
                subprocess.run([sys.executable,
                                os.path.join(HERE, "plot_inversion.py"),
                                "--dir", res["out_dir"]], check=False)
    print("\nSummary")
    for sid, cfg, res in summary:
        if res is None:
            print(f"  seg{sid:02d}/{cfg}: FAILED")
        else:
            print(f"  seg{sid:02d}/{cfg}: rmse_med={res['rmse_median']:.1f} m  "
                  f"wall={res['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
