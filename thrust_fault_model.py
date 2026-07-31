"""
thrust_fault_model.py

Core (non-plotting) functions for:
- reading + ordering a river profile
- optional Savitzky-Golay smoothing of many CSVs
- C2-continuous piecewise-linear-with-quintic-patches fault geometry
- structural uplift along a master distance axis
- stream-power forward model + K fitting
- objective (chi^2) and log-posterior (for emcee)

Designed so that *all parameters* live in your notebook and get passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ArrayLike = Union[np.ndarray, Sequence[float]]


# -----------------------------------------------------------------------------
# I/O helpers (no plotting)
# -----------------------------------------------------------------------------

def wfix(n: int, w: int) -> int:
    """
    Ensure Savitzky-Golay window is:
      - <= n-1 (if n odd) or <= n-2 (if n even)
      - odd
    """
    if n < 3:
        raise ValueError("wfix: need n >= 3")
    w = min(int(w), n - (1 if n % 2 else 2))
    return w + (w % 2 == 0)


def smooth_river_csvs(
    in_dir: str,
    out_dir: str,
    *,
    w: int = 211,
    p: int = 1,
    columns: Tuple[str, str, str] = ("sim_x", "sim_y", "elevation"),
) -> List[str]:
    """
    Apply Savitzky-Golay smoothing to the elevation column for all CSVs in in_dir,
    writing the result to out_dir. Returns list of written filenames.

    This reproduces your "batch smooth all files" step without any plotting.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    csvs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found in: {in_dir}")

    cx, cy, cz = columns

    for fn in csvs:
        path = os.path.join(in_dir, fn)
        df = pd.read_csv(path)

        if not {cx, cy, cz} <= set(df.columns) or len(df) < 3:
            # silently skip non-conforming files (same spirit as your script)
            continue

        z = df[cz].to_numpy(float)
        ww = wfix(len(z), w)

        df2 = df.copy()
        df2[cz] = savgol_filter(z, ww, p)
        out_path = os.path.join(out_dir, fn)
        df2.to_csv(out_path, index=False)

        written.append(fn)

    return written


def cumulative_arclength(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Cumulative Euclidean arclength (same units as x,y). First element is 0.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate(([0.0], np.cumsum(ds)))


def interp_linear_extrap(xq: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    1D linear interpolation with linear extrapolation on both ends.
    Requires x strictly increasing.
    """
    xq = np.asarray(xq, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("interp_linear_extrap: x must be strictly increasing")

    yq = np.interp(xq, x, y)

    left = xq < x[0]
    if np.any(left):
        mL = (y[1] - y[0]) / (x[1] - x[0])
        yq[left] = y[0] + mL * (xq[left] - x[0])

    right = xq > x[-1]
    if np.any(right):
        mR = (y[-1] - y[-2]) / (x[-1] - x[-2])
        yq[right] = y[-1] + mR * (xq[right] - x[-1])

    return yq


def length_weights_from_s(s_km: ArrayLike) -> np.ndarray:
    """
    Length-based node weights from an along-channel coordinate s_km.
    Normalized so mean weight = 1.

    Matches your existing implementation.
    """
    s_km = np.asarray(s_km, float)
    s_m = (s_km - s_km[0]) * 1000.0
    ds = np.diff(s_m)

    w = np.empty_like(s_m)
    w[0] = ds[0]
    w[-1] = ds[-1]
    w[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    w = w / np.mean(w)
    return w


# -----------------------------------------------------------------------------
# Stream-power pieces (no plotting)
# -----------------------------------------------------------------------------

def precompute_streampower_geometry_terms(
    x_km: ArrayLike,
    *,
    hack_exp: float,
    l0_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute:
      - ds_m: segment lengths (meters) between adjacent x_km nodes
      - A_mid: midpoint drainage-area proxy on segments

    Assumes x_km is along-channel distance from outlet (in km).
    """
    x_km = np.asarray(x_km, float)

    s_m = (x_km - x_km[0]) * 1000.0
    ds_m = np.diff(s_m)

    L_m = float(s_m[-1])
    l_to_divide_m = L_m - s_m

    A_proxy = (l_to_divide_m + float(l0_m)) ** float(hack_exp)
    A_mid = 0.5 * (A_proxy[:-1] + A_proxy[1:])
    return ds_m, A_mid


def stream_power_integral_I_from_uplift(
    uplift_mm_yr: ArrayLike,
    ds_m: ArrayLike,
    A_mid: ArrayLike,
    *,
    m_sp: float,
    n_sp: float,
) -> np.ndarray:
    """
    Compute stream-power integral I(s) from uplift.
    Negative uplift is floored to 0 (as in your code).
    """
    U_m_yr = np.maximum(np.asarray(uplift_mm_yr, float) / 1000.0, 0.0)
    ds_m = np.asarray(ds_m, float)
    A_mid = np.asarray(A_mid, float)

    U_mid = 0.5 * (U_m_yr[:-1] + U_m_yr[1:])

    integrand = np.zeros_like(U_mid)
    mask = U_mid > 0.0
    if np.any(mask):
        integrand[mask] = (U_mid[mask] / (A_mid[mask] ** float(m_sp))) ** (1.0 / float(n_sp))

    I = np.concatenate(([0.0], np.cumsum(integrand * ds_m)))
    return I


def fit_K_given_baselevel(I: ArrayLike, z_obs: ArrayLike, *, n_sp: float) -> Tuple[float, np.ndarray, float]:
    """
    Given I(s) and observed elevation z_obs(s), fit K (and baselevel z0) via
    least squares on z = z0 + a * I, then K = a^(-n).

    Returns (K_fit, z_pred, z0). If invalid, returns (nan, nan-array, nan).
    """
    I = np.asarray(I, float)
    z_obs = np.asarray(z_obs, float)

    z0 = float(z_obs[0])
    y = z_obs - z0
    denom = float(np.dot(I, I))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, np.full_like(z_obs, np.nan), np.nan

    a = float(np.dot(I, y) / denom)
    if a <= 0 or not np.isfinite(a):
        return np.nan, np.full_like(z_obs, np.nan), z0

    K = a ** (-float(n_sp))
    z_pred = z0 + a * I
    return K, z_pred, z0


# -----------------------------------------------------------------------------
# Fault geometry (C2 continuous model)
# -----------------------------------------------------------------------------

def _quintic_patch_coeffs(xL: float, zL: float, mL: float, xR: float, zR: float, mR: float) -> Tuple[np.ndarray, float]:
    """
    Quintic patch with:
      z(xL)=zL, z'(xL)=mL, z''(xL)=0
      z(xR)=zR, z'(xR)=mR, z''(xR)=0
    using normalized variable t in [0,1].

    Returns (coeffs, h) where h = xR-xL and coeffs are in t-space.
    """
    h = xR - xL
    if h <= 0:
        raise ValueError("xR must be > xL")

    a0 = zL
    a1 = mL * h
    a2 = 0.0

    D0 = zR - zL - a1
    D1 = (mR - mL) * h

    A = np.array([[1, 1, 1],
                  [3, 4, 5],
                  [6, 12, 20]], dtype=float)

    a3, a4, a5 = np.linalg.solve(A, np.array([D0, D1, 0.0], dtype=float))
    return np.array([a0, a1, a2, a3, a4, a5], dtype=float), h


def _eval_quintic(x: np.ndarray, xL: float, coeffs: np.ndarray, h: float) -> np.ndarray:
    t = (x - xL) / h
    a0, a1, a2, a3, a4, a5 = coeffs
    return (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t + a0)


def build_c2_fault_model(x: ArrayLike, z: ArrayLike, *, w: float = 5.0) -> Dict[str, object]:
    """
    Build a piecewise model that is linear on segments but replaces each node
    junction with a quintic patch of half-width w_i to enforce C2 continuity.

    Returns a dict model used by eval_c2_fault().
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points")

    m_seg = np.diff(z) / np.diff(x)

    wv = np.zeros(n)
    for i in range(1, n - 1):
        wi = float(w)
        wi = min(wi, 0.45 * (x[i] - x[i - 1]), 0.45 * (x[i + 1] - x[i]))
        wv[i] = wi

    patches: Dict[int, Tuple[float, np.ndarray, float]] = {}
    for i in range(1, n - 1):
        wi = wv[i]
        if wi <= 0:
            continue

        xL = x[i] - wi
        xR = x[i] + wi
        mL = m_seg[i - 1]
        mR = m_seg[i]
        zL = z[i] + mL * (xL - x[i])
        zR = z[i] + mR * (xR - x[i])

        coeffs, h = _quintic_patch_coeffs(xL, zL, mL, xR, zR, mR)
        patches[i] = (xL, coeffs, h)

    pieces: List[Tuple[str, float, float, int]] = []
    for seg in range(n - 1):
        left_cut = wv[seg] if seg > 0 else 0.0
        right_cut = wv[seg + 1] if (seg + 1) < n - 1 else 0.0

        xL = x[seg] + left_cut
        xR = x[seg + 1] - right_cut
        if xR > xL:
            pieces.append(("line", xL, xR, seg))

        v = seg + 1
        if 1 <= v <= n - 2 and wv[v] > 0:
            pieces.append(("patch", x[v] - wv[v], x[v] + wv[v], v))

    return dict(x=x, z=z, m_seg=m_seg, pieces=pieces, patches=patches)


def eval_c2_fault(xq: ArrayLike, model: Mapping[str, object], *, extrapolate: bool = True) -> np.ndarray:
    """
    Evaluate the C2 fault model at query points xq.
    """
    x = np.asarray(model["x"], float)
    z = np.asarray(model["z"], float)
    m_seg = np.asarray(model["m_seg"], float)
    pieces = list(model["pieces"])  # type: ignore
    patches = dict(model["patches"])  # type: ignore

    xq = np.asarray(xq, float)
    zq = np.full_like(xq, np.nan, dtype=float)

    for kind, xL, xR, idx in pieces:
        mask = (xq >= xL) & (xq <= xR)
        if not np.any(mask):
            continue

        if kind == "line":
            seg = idx
            zq[mask] = z[seg] + m_seg[seg] * (xq[mask] - x[seg])
        else:
            v = idx
            xL0, coeffs, h = patches[v]
            zq[mask] = _eval_quintic(xq[mask], xL0, coeffs, h)

    if extrapolate:
        m0 = m_seg[0]
        mn = m_seg[-1]
        zq = np.where(np.isnan(zq) & (xq < x[0]), z[0] + m0 * (xq - x[0]), zq)
        zq = np.where(np.isnan(zq) & (xq > x[-1]), z[-1] + mn * (xq - x[-1]), zq)

    return zq


def resample_equal_arclength(x_dense: ArrayLike, z_dense: ArrayLike, n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Resample a polyline (x_dense, z_dense) to n_points equally spaced in arclength.
    Returns (x_new, z_new, s_uniform, total_length).
    """
    x_dense = np.asarray(x_dense, float)
    z_dense = np.asarray(z_dense, float)

    ds = np.sqrt(np.diff(x_dense) ** 2 + np.diff(z_dense) ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    L = float(s[-1])

    s_uniform = np.linspace(0.0, L, int(n_points))
    x_new = np.interp(s_uniform, s, x_dense)
    z_new = np.interp(s_uniform, s, z_dense)

    return x_new, z_new, s_uniform, L


def segment_angles(xnode: ArrayLike, znode: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment angles theta (radians), dx, dz for consecutive node pairs.
    """
    xnode = np.asarray(xnode, dtype=float)
    znode = np.asarray(znode, dtype=float)

    dx = np.diff(xnode)
    dz = np.diff(znode)
    theta = np.arctan2(dz, dx)
    return theta, dx, dz


def compute_axial_surface_intersections(
    xnode: ArrayLike,
    znode: ArrayLike,
    theta: ArrayLike,
    *,
    atol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute x-intercepts of axial planes with the surface (z=0), and gamma angles.
    Raises if resulting x_axial (finite values) is not strictly increasing.
    """
    xnode = np.asarray(xnode, dtype=float)
    znode = np.asarray(znode, dtype=float)
    theta = np.asarray(theta, dtype=float)

    nseg = theta.size
    if nseg < 2:
        raise ValueError("Need at least 2 segments to compute axial planes")

    x_axial = np.empty(nseg - 1, dtype=float)
    gamma = np.empty(nseg - 1, dtype=float)

    for j in range(1, nseg):
        th1 = theta[j - 1]
        th2 = theta[j]
        g = 0.5 * (th1 + th2 + np.pi)
        gamma[j - 1] = g

        xk, zk = xnode[j], znode[j]
        tg = np.tan(g)

        if np.isclose(tg, 0.0, atol=atol):
            x_axial[j - 1] = np.nan
        else:
            x_axial[j - 1] = xk - zk / tg

    x_bounds = x_axial[np.isfinite(x_axial)]
    bad = np.where(np.diff(x_bounds) <= 0)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"x_axial is not strictly increasing at i={i}: {x_bounds[i]} -> {x_bounds[i+1]}")

    return x_axial, gamma


def structural_velocity(
    x_obs: ArrayLike,
    theta: ArrayLike,
    x_axial: ArrayLike,
    slip_by_segment: ArrayLike,
    *,
    x_start: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (u,v) structural velocity components at x_obs (same axis as x_axial),
    for a set of segment angles theta and slip_by_segment.

    Returns (u, v, dom_index).
    """
    x_obs = np.asarray(x_obs, dtype=float)
    theta = np.asarray(theta, dtype=float)
    slip_by_segment = np.asarray(slip_by_segment, dtype=float)
    x_axial = np.asarray(x_axial, dtype=float)

    x_bounds = x_axial[np.isfinite(x_axial)]
    dom = np.searchsorted(x_bounds, x_obs, side="right")
    dom = np.clip(dom, 0, theta.size - 1)

    u = slip_by_segment[dom] * np.cos(theta[dom])
    v = -slip_by_segment[dom] * np.sin(theta[dom])

    mask = x_obs < float(x_start)
    u = u.astype(float, copy=True)
    v = v.astype(float, copy=True)
    u[mask] = 0.0
    v[mask] = 0.0

    return u, v, dom


def dips_ok(
    xctrl: ArrayLike,
    zctrl: ArrayLike,
    *,
    dip_min_deg: float = 5.0,
    dip_max_deg: float = 80.0,
) -> bool:
    """
    True if all segment dips are within (dip_min_deg, dip_max_deg), where dip is
    computed as arctan(-dz/dx) in degrees.
    """
    xctrl = np.asarray(xctrl, float)
    zctrl = np.asarray(zctrl, float)
    dips = np.degrees(np.arctan2(-np.diff(zctrl), np.diff(xctrl)))
    return bool(np.all((dips > float(dip_min_deg)) & (dips < float(dip_max_deg))))


# -----------------------------------------------------------------------------
# Fault parameterization (q + log-depth increments)
# -----------------------------------------------------------------------------

def _norm_idx(i: int, n: int) -> int:
    i = int(i)
    return i if i >= 0 else n + i


def _coerce_fixed(d: Optional[Mapping[int, float]], n: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if d is None:
        return out
    for k, v in dict(d).items():
        kk = _norm_idx(int(k), n)
        if not (0 <= kk < n):
            raise ValueError(f"fixed index {k} -> {kk} out of range for n_ctrl={n}")
        out[kk] = float(v)
    return out


def fault_free_x_indices(cfg: Mapping[str, object]) -> List[int]:
    """
    Indices (in ctrl-node list) where x is free (parameterized by q's).
    """
    n = int(cfg["n_ctrl"])
    x_end = float(cfg["x_end_km"])

    x_fixed = _coerce_fixed(cfg.get("x_fixed", {}) if isinstance(cfg, dict) else {}, n)
    x_fixed.setdefault(0, 0.0)
    x_fixed.setdefault(n - 1, x_end)

    fixed = sorted(x_fixed.keys())
    q_order: List[int] = []
    for a, b in zip(fixed[:-1], fixed[1:]):
        for i in range(a + 1, b):
            if i not in x_fixed:
                q_order.append(i)
    return q_order


def fault_param_counts(cfg: Mapping[str, object]) -> Tuple[int, int]:
    """
    Returns (n_q, n_ld) where:
      - q's parameterize free x positions
      - ld's are log depth increments (n_ctrl-1 of them)
    """
    n = int(cfg["n_ctrl"])
    n_q = len(fault_free_x_indices(cfg))
    n_ld = n - 1
    return n_q, n_ld


def params_to_fault_ctrl(p: ArrayLike, *, cfg: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map unconstrained parameters to (xctrl, zctrl).

    p = [q_1..q_nq, ld_1..ld_nld]
      - q in (0,1) place internal x nodes within fixed-anchor intervals
      - ld are log depth increments; depth increments are exp(ld) and then
        rescaled to satisfy any z_fixed anchors.
    """
    n = int(cfg["n_ctrl"])
    x_end = float(cfg["x_end_km"])

    x_fixed = _coerce_fixed(cfg.get("x_fixed", {}) if isinstance(cfg, dict) else {}, n)
    x_fixed.setdefault(0, 0.0)
    x_fixed.setdefault(n - 1, x_end)

    z_fixed = _coerce_fixed(cfg.get("z_fixed", {}) if isinstance(cfg, dict) else {}, n)
    z_fixed.setdefault(0, 0.0)

    q_order = fault_free_x_indices(cfg)
    n_q, n_ld = fault_param_counts(cfg)

    p = np.asarray(p, float)
    if p.size != n_q + n_ld:
        raise ValueError(f"Expected {n_q + n_ld} fault params, got {p.size}")

    q = p[:n_q]
    ld = p[n_q:]
    d_raw = np.exp(ld)

    # --- build xctrl
    x = np.full(n, np.nan, float)
    for i, v in x_fixed.items():
        x[i] = float(v)

    fixed_idx = sorted(x_fixed.keys())
    qi = 0
    for a, b in zip(fixed_idx[:-1], fixed_idx[1:]):
        xL, xR = float(x[a]), float(x[b])
        if not (xL < xR):
            raise ValueError(f"x_fixed must increase with node index: idx {a}->{b} gives {xL}->{xR}")

        xx = xL
        for i in range(a + 1, b):
            if i in x_fixed:
                continue
            qq = float(q[qi]); qi += 1
            xx = xx + (xR - xx) * qq
            x[i] = xx

    if qi != n_q or np.any(~np.isfinite(x)):
        raise RuntimeError("xctrl build failed")

    # --- build zctrl with anchors
    if 0 not in z_fixed:
        raise ValueError("z_fixed must include node 0")

    anchors = sorted(z_fixed.keys())
    if anchors[0] != 0:
        raise ValueError("z_fixed must include node 0 as the first anchor")

    for ia, ib in zip(anchors[:-1], anchors[1:]):
        if ib <= ia:
            raise ValueError("z_fixed indices must be strictly increasing")
        if not (z_fixed[ib] < z_fixed[ia]):
            raise ValueError(
                f"z_fixed must get deeper with node index: z[{ia}]={z_fixed[ia]} vs z[{ib}]={z_fixed[ib]}"
            )

    z = np.full(n, np.nan, float)
    z[0] = float(z_fixed[0])

    last = 0
    for nxt in anchors[1:]:
        z_last = float(z[last])
        z_nxt = float(z_fixed[nxt])
        total = z_last - z_nxt
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(f"Bad z_fixed interval {last}->{nxt}: {z_last}->{z_nxt}")

        seg_raw = d_raw[last:nxt]
        ssum = float(np.sum(seg_raw))
        if ssum <= 0.0 or not np.isfinite(ssum):
            raise ValueError("Bad depth increment sum")

        seg = seg_raw / ssum * total
        for k in range(last, nxt):
            z[k + 1] = z[k] - seg[k - last]
        z[nxt] = z_nxt
        last = nxt

    for k in range(last, n - 1):
        z[k + 1] = z[k] - d_raw[k]

    if np.any(~np.isfinite(z)):
        raise RuntimeError("zctrl build failed")

    return x, z


def fault_bounds_for_de(
    *,
    cfg: Mapping[str, object],
    eps: float = 0.01,
    ld_lo: float = np.log(0.5),
    ld_hi: float = np.log(40.0),
) -> List[Tuple[float, float]]:
    """
    Bounds for differential evolution on the *fault* parameters only.
    """
    n_q, n_ld = fault_param_counts(cfg)
    return [(eps, 1.0 - eps)] * n_q + [(ld_lo, ld_hi)] * n_ld


# -----------------------------------------------------------------------------
# Data bundle + forward model
# -----------------------------------------------------------------------------

@dataclass
class ChannelData:
    """
    Everything the forward model needs about the observed channel profile.
    """
    X: np.ndarray
    Y: np.ndarray
    Zobs: np.ndarray

    # ordered from outlet
    s_km_model: np.ndarray
    Zobs_order: np.ndarray
    x_uplift_coord_order: np.ndarray

    # weights for chi^2
    w_node: np.ndarray

    # precomputed stream-power geometry
    ds_m: np.ndarray
    A_mid: np.ndarray

    outlet_is_start: bool
    order: np.ndarray


def load_channel_profile_csv(
    file_path: str,
    *,
    uplift_sample_axis: str,
    use_length_weights: bool,
    hack_exp: float,
    l0_m: float,
    columns: Tuple[str, str, str] = ("sim_x", "sim_y", "elevation"),
) -> ChannelData:
    """
    Load a single profile CSV and construct ChannelData.

    Parameters
    ----------
    uplift_sample_axis
        'sim_x' or 'sim_y'. This chooses which coordinate you want to use
        to sample uplift from x_master via interpolation.

    hack_exp, l0_m
        Stream-power area proxy parameters.
    """
    cx, cy, cz = columns
    df = pd.read_csv(file_path)

    if not {cx, cy, cz} <= set(df.columns):
        raise ValueError(f"CSV must contain columns {columns}. Found {tuple(df.columns)}")

    X = df[cx].to_numpy(float)
    Y = df[cy].to_numpy(float)
    Zobs = df[cz].to_numpy(float)

    if uplift_sample_axis == cy:
        x_uplift_coord = Y
    elif uplift_sample_axis == cx:
        x_uplift_coord = X
    else:
        raise ValueError(f"uplift_sample_axis must be '{cx}' or '{cy}'")

    s = cumulative_arclength(X, Y)
    outlet_is_start = bool(Zobs[0] <= Zobs[-1])
    order = np.arange(len(X)) if outlet_is_start else np.arange(len(X) - 1, -1, -1)

    s_km_model = (s if outlet_is_start else (s[-1] - s))[order]
    Zobs_order = Zobs[order]
    x_uplift_coord_order = x_uplift_coord[order]

    w_node = length_weights_from_s(s_km_model) if use_length_weights else np.ones_like(Zobs_order)

    ds_m, A_mid = precompute_streampower_geometry_terms(
        s_km_model, hack_exp=hack_exp, l0_m=l0_m
    )

    return ChannelData(
        X=X,
        Y=Y,
        Zobs=Zobs,
        s_km_model=s_km_model,
        Zobs_order=Zobs_order,
        x_uplift_coord_order=x_uplift_coord_order,
        w_node=w_node,
        ds_m=ds_m,
        A_mid=A_mid,
        outlet_is_start=outlet_is_start,
        order=order,
    )


def compute_uplift_from_fault(
    xctrl: ArrayLike,
    zctrl: ArrayLike,
    *,
    x_master: ArrayLike,
    structuralslip: float,
    w_blend: float,
    n_dense: int = 600,
    n_nodes: int = 350,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    From a fault polyline (xctrl, zctrl), compute vertical velocity v(x_master)
    using the structural geometry method in your script.

    Returns:
      uplift_mm_yr (same length as x_master), and (xnode, znode) used internally.
    """
    x_master = np.asarray(x_master, float)

    m = build_c2_fault_model(xctrl, zctrl, w=w_blend)
    xd = np.linspace(float(np.min(xctrl)), float(np.max(xctrl)), int(n_dense))
    zd = eval_c2_fault(xd, m)

    xn, zn, _, _ = resample_equal_arclength(xd, zd, int(n_nodes))
    th, _, _ = segment_angles(xn, zn)

    xa, _ = compute_axial_surface_intersections(xn, zn, th)
    slip_by_seg = float(structuralslip) * np.ones_like(th)

    _, v, _ = structural_velocity(
        x_master, th, xa, slip_by_seg, x_start=float(xn[0])
    )

    return v, (xn, zn)


def forward_model_onepass(
    p_fault: ArrayLike,
    *,
    data: ChannelData,
    fault_cfg: Mapping[str, object],
    x_master: ArrayLike,
    structuralslip: float,
    w_blend: float,
    m_sp: float,
    n_sp: float,
    n_dense: int,
    n_nodes: int,
) -> Optional[Dict[str, object]]:
    """
    One forward-model evaluation.

    Returns dict with:
      xctrl, zctrl, uplift_mm_yr, uplift_on_channel, I, K_fit, z_pred, z0, xnode, znode
    or None if constraints fail / numerical errors.
    """
    try:
        xctrl, zctrl = params_to_fault_ctrl(p_fault, cfg=fault_cfg)
    except Exception:
        return None

    # hard constraints
    if np.min(np.diff(xctrl)) < float(fault_cfg["dx_min_km"]):
        return None

    if not dips_ok(
        xctrl, zctrl,
        dip_min_deg=float(fault_cfg["dip_min_deg"]),
        dip_max_deg=float(fault_cfg["dip_max_deg"]),
    ):
        return None

    if np.min(zctrl) < float(fault_cfg["zmin_ctrl_km"]):
        return None

    try:
        uplift_mm_yr, (xnode, znode) = compute_uplift_from_fault(
            xctrl, zctrl,
            x_master=x_master,
            structuralslip=structuralslip,
            w_blend=w_blend,
            n_dense=n_dense,
            n_nodes=n_nodes,
        )
    except Exception:
        return None

    if np.min(znode) < float(fault_cfg["zmin_node_km"]):
        return None

    uplift_on_channel = interp_linear_extrap(
        data.x_uplift_coord_order,
        np.asarray(x_master, float),
        uplift_mm_yr
    )

    I = stream_power_integral_I_from_uplift(
        uplift_on_channel,
        data.ds_m,
        data.A_mid,
        m_sp=float(m_sp),
        n_sp=float(n_sp),
    )

    K_fit, z_pred, z0 = fit_K_given_baselevel(I, data.Zobs_order, n_sp=float(n_sp))
    if not np.isfinite(K_fit):
        return None

    return dict(
        xctrl=xctrl,
        zctrl=zctrl,
        uplift_mm_yr=uplift_mm_yr,
        uplift_on_channel=uplift_on_channel,
        I=I,
        K_fit=float(K_fit),
        z_pred=z_pred,
        z0=float(z0),
        xnode=xnode,
        znode=znode,
    )


# -----------------------------------------------------------------------------
# Objective (for DE) and Posterior (for emcee)
# -----------------------------------------------------------------------------

@dataclass
class DEObjective:
    """
    Chi^2 objective for optimizing fault geometry only (with fixed m_sp, n_sp).
    """
    data: ChannelData
    fault_cfg: Mapping[str, object]
    x_master: np.ndarray
    structuralslip: float
    w_blend: float
    sigma_z_m: float
    m_sp: float
    n_sp: float
    n_dense: int = 600
    n_nodes: int = 350

    def __call__(self, p_fault: ArrayLike) -> float:
        out = forward_model_onepass(
            p_fault,
            data=self.data,
            fault_cfg=self.fault_cfg,
            x_master=self.x_master,
            structuralslip=self.structuralslip,
            w_blend=self.w_blend,
            m_sp=float(self.m_sp),
            n_sp=float(self.n_sp),
            n_dense=int(self.n_dense),
            n_nodes=int(self.n_nodes),
        )
        if out is None:
            return 1e50

        r = self.data.Zobs_order - out["z_pred"]
        val = float(np.sum(self.data.w_node * (r / float(self.sigma_z_m)) ** 2))

        # soft end-depth penalty (optional keys)
        z_soft = self.fault_cfg.get("z_end_soft_km", None) if isinstance(self.fault_cfg, dict) else None
        if z_soft is not None:
            z_end = float(out["zctrl"][-1])
            wt = float(self.fault_cfg.get("z_end_soft_wt", 0.0))  # type: ignore
            val = float(val + wt * max(0.0, float(z_soft) - z_end) ** 2)

        return val


@dataclass
class Posterior:
    """
    Log-posterior for theta = [p_fault..., m_sp, n_sp].

    All bounds are supplied from the notebook so you can control eps, mlo/mhi, etc.
    """
    data: ChannelData
    fault_cfg: Mapping[str, object]
    x_master: np.ndarray
    structuralslip: float
    w_blend: float
    sigma_z_m: float

    # priors / bounds (set in notebook)
    eps: float
    ld_lo: float
    ld_hi: float
    mlo: float
    mhi: float
    nlo: float
    nhi: float

    # forward model resolution used inside likelihood
    n_dense_like: int = 600
    n_nodes_like: int = 350

    def __post_init__(self) -> None:
        n_q, n_ld = fault_param_counts(self.fault_cfg)
        self.n_q = int(n_q)
        self.n_ld = int(n_ld)
        self.n_fault_params = int(n_q + n_ld)
        self.ndim = int(self.n_fault_params + 2)

    def split_theta(self, th: ArrayLike) -> Tuple[np.ndarray, float, float]:
        th = np.asarray(th, float)
        if th.size != self.ndim:
            raise ValueError(f"Expected theta size {self.ndim}, got {th.size}")
        p_fault = th[: self.n_fault_params]
        mr = float(th[self.n_fault_params])
        nr = float(th[self.n_fault_params + 1])
        return p_fault, mr, nr

    def log_prior(self, th: ArrayLike) -> float:
        p_fault, mr, nr = self.split_theta(th)

        q = p_fault[: self.n_q]
        ld = p_fault[self.n_q :]

        eps = float(self.eps)
        if np.any(q <= eps) or np.any(q >= 1.0 - eps):
            return -np.inf
        if np.any(ld <= float(self.ld_lo)) or np.any(ld >= float(self.ld_hi)):
            return -np.inf
        if not (float(self.mlo) <= mr <= float(self.mhi)):
            return -np.inf
        if not (float(self.nlo) <= nr <= float(self.nhi)):
            return -np.inf

        # geometry constraints
        try:
            x, z = params_to_fault_ctrl(p_fault, cfg=self.fault_cfg)
        except Exception:
            return -np.inf

        if np.min(np.diff(x)) < float(self.fault_cfg["dx_min_km"]):
            return -np.inf

        if not dips_ok(
            x, z,
            dip_min_deg=float(self.fault_cfg["dip_min_deg"]),
            dip_max_deg=float(self.fault_cfg["dip_max_deg"]),
        ):
            return -np.inf

        if np.min(z) < float(self.fault_cfg["zmin_ctrl_km"]):
            return -np.inf

        return 0.0

    def log_likelihood(self, th: ArrayLike) -> float:
        p_fault, mr, nr = self.split_theta(th)

        out = forward_model_onepass(
            p_fault,
            data=self.data,
            fault_cfg=self.fault_cfg,
            x_master=self.x_master,
            structuralslip=self.structuralslip,
            w_blend=self.w_blend,
            m_sp=float(mr),
            n_sp=float(nr),
            n_dense=int(self.n_dense_like),
            n_nodes=int(self.n_nodes_like),
        )
        if out is None:
            return -np.inf

        r = self.data.Zobs_order - out["z_pred"]
        chi2 = float(np.sum(self.data.w_node * (r / float(self.sigma_z_m)) ** 2))
        return -0.5 * chi2

    def __call__(self, th: ArrayLike) -> float:
        lp = self.log_prior(th)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(th)
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)

    def draw_valid(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw one prior sample that passes log-posterior finite check.
        """
        while True:
            p_fault = np.r_[
                rng.uniform(float(self.eps), 1.0 - float(self.eps), size=self.n_q),
                rng.uniform(float(self.ld_lo), float(self.ld_hi), size=self.n_ld),
            ].astype(float)

            mr = float(rng.uniform(float(self.mlo), float(self.mhi)))
            nr = float(rng.uniform(float(self.nlo), float(self.nhi)))
            th = np.r_[p_fault, mr, nr].astype(float)

            if np.isfinite(self(th)):
                return th

    def evaluate(
        self,
        th: ArrayLike,
        *,
        n_dense: int = 1000,
        n_nodes: int = 1000,
    ) -> Optional[Dict[str, object]]:
        """
        Convenience wrapper to run the forward model for a specific theta.
        """
        p_fault, mr, nr = self.split_theta(th)
        return forward_model_onepass(
            p_fault,
            data=self.data,
            fault_cfg=self.fault_cfg,
            x_master=self.x_master,
            structuralslip=self.structuralslip,
            w_blend=self.w_blend,
            m_sp=float(mr),
            n_sp=float(nr),
            n_dense=int(n_dense),
            n_nodes=int(n_nodes),
        )
