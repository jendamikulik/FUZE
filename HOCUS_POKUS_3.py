#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#D:/HIT/PythonProject/.venv/Scripts/python HOCUS_POKUS_3.py --core-summary core_summary.json --spectrum fig2c_data.mat --dataset-key "Sample A, 5$\mathcal{\times10^{13} cm^{-2}}$" --out-prefix fuze
#D:/HIT/PythonProject/.venv/Scripts/python .\HOCUS_POKUS_3.py --core-summary core_summary.json --spectrum fig2c_data.mat --dataset-key "Sample A, 5$\mathcal{\times10^{13} cm^{-2}}$" --ringdown ringdown.csv --ringdown-xcol t --ringdown-ycol alpha --closure closure.csv --closure-xcol t --closure-ycol T --out-prefix final


from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares

try:
    import h5py  # type: ignore
except Exception:
    h5py = None


# ============================================================
# Small utilities
# ============================================================

def robust_mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def rmse(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_fit) ** 2)))


def bic_score(y_true: np.ndarray, y_fit: np.ndarray, k: int) -> float:
    n = len(y_true)
    rss = float(np.sum((y_true - y_fit) ** 2))
    return float(n * np.log(rss / max(n, 1) + 1e-300) + k * np.log(max(n, 1)))


def aic_score(y_true: np.ndarray, y_fit: np.ndarray, k: int) -> float:
    n = len(y_true)
    rss = float(np.sum((y_true - y_fit) ** 2))
    return float(n * np.log(rss / max(n, 1) + 1e-300) + 2 * k)


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return 0.0
    a = a[m]
    b = b[m]
    sa = np.std(a)
    sb = np.std(b)
    if sa <= 1e-15 or sb <= 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ensure_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    return x[idx], y[idx]


def smooth1d(y: np.ndarray, window: int = 9) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def canon_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("\\", "/")
        .replace(" ", "")
        .replace("_", "")
    )


# ============================================================
# Safe summary parsing
# ============================================================

@dataclass
class PeakSeed:
    A: float
    x0: float
    gamma: float
    fwhm: float

    @property
    def Q(self) -> float:
        return float(self.x0 / max(self.fwhm, 1e-12))


@dataclass
class SafeConfig:
    data_path: Optional[str]
    xcol: Optional[str]
    ycol: Optional[str]
    fit_window: Tuple[Optional[float], Optional[float]]
    core_window: Tuple[float, float]
    baseline_degree: int
    sigma_noise: float
    transfer_supported: bool
    seed_peaks: List[PeakSeed]
    primary_peaks: List[PeakSeed]




# ============================================================
# MAT / CSV loading with exact xcol / ycol matching
# ============================================================

def matlab_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [matlab_to_python(v) for v in obj.flat]
        return obj
    return obj


def collect_numeric_candidates(obj: Any, prefix: str = "root") -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []

    def _walk(o: Any, name: str) -> None:
        if o is None:
            return

        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).startswith("__"):
                    continue
                _walk(v, f"{name}.{k}")
            return

        if isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f"{name}[{i}]")
            return

        if hasattr(o, "_fieldnames"):
            for fn in getattr(o, "_fieldnames", []):
                _walk(getattr(o, fn), f"{name}.{fn}")
            return

        if isinstance(o, np.ndarray):
            if o.dtype.names:
                for fn in o.dtype.names:
                    try:
                        _walk(o[fn], f"{name}.{fn}")
                    except Exception:
                        pass
                return
            if o.dtype == object:
                for i, v in enumerate(o.flat):
                    _walk(v, f"{name}[{i}]")
                return
            if np.issubdtype(o.dtype, np.number):
                arr = np.squeeze(np.asarray(o))
                if arr.ndim == 1 and arr.size >= 8:
                    out.append((name, arr.astype(float)))
            return

        try:
            arr = np.asarray(o)
        except Exception:
            return

        if arr.dtype == object:
            for i, v in enumerate(arr.flat):
                _walk(v, f"{name}[{i}]")
            return

        if np.issubdtype(arr.dtype, np.number):
            arr = np.squeeze(arr)
            if arr.ndim == 1 and arr.size >= 8:
                out.append((name, arr.astype(float)))

    _walk(obj, prefix)

    uniq = []
    seen = set()
    for name, arr in out:
        key = (name, tuple(arr.shape))
        if key not in seen:
            seen.add(key)
            uniq.append((name, arr))
    return uniq


def collect_hdf5_candidates(path: Path) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    if h5py is None:
        return out
    try:
        with h5py.File(path, "r") as f:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = np.array(obj)
                        arr = np.squeeze(arr)
                        if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1 and arr.size >= 8:
                            out.append((f"{path.stem}.{name.replace('/', '.')}", arr.astype(float)))
                    except Exception:
                        pass
            f.visititems(visitor)
    except Exception:
        pass
    return out


def choose_exact_candidate(
    candidates: Sequence[Tuple[str, np.ndarray]],
    target_name: str,
) -> Tuple[str, np.ndarray]:
    if not target_name:
        raise RuntimeError("Missing target column name for exact candidate lookup.")

    target_c = canon_name(target_name)

    # exact canonical match first
    for name, arr in candidates:
        if canon_name(name) == target_c:
            return name, arr

    # suffix match
    for name, arr in candidates:
        cn = canon_name(name)
        if cn.endswith(target_c) or target_c.endswith(cn):
            return name, arr

    raise RuntimeError(
        f"Exact MAT candidate not found for '{target_name}'.\n"
        f"Available candidates (first 80):\n" + "\n".join(name for name, _ in candidates[:80])
    )


def load_xy_mat_exact(path: Path, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    errors = []
    candidates: List[Tuple[str, np.ndarray]] = []

    try:
        data = loadmat(path, simplify_cells=True)
        data = {k: matlab_to_python(v) for k, v in data.items() if not str(k).startswith("__")}
        candidates.extend(collect_numeric_candidates(data, prefix=path.stem))
    except Exception as e:
        errors.append(f"loadmat: {e}")

    if not candidates:
        try:
            candidates.extend(collect_hdf5_candidates(path))
        except Exception as e:
            errors.append(f"h5py: {e}")

    if not candidates:
        raise RuntimeError(
            "MAT exact loader found no numeric candidates. Details: " + "; ".join(errors)
        )

    xname, x = choose_exact_candidate(candidates, xcol)
    yname, y = choose_exact_candidate(candidates, ycol)

    if len(x) != len(y):
        raise RuntimeError(
            f"Exact MAT columns have different lengths: len(x)={len(x)}, len(y)={len(y)}"
        )

    x, y = ensure_sorted_xy(np.asarray(x, float), np.asarray(y, float))
    return x, y, xname, yname


def load_xy_csv_exact(path: Path, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt", ".dat"}:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if xcol in df.columns and ycol in df.columns:
        xc, yc = xcol, ycol
    else:
        # fallback by canonical name
        cmap = {canon_name(c): c for c in df.columns}
        xc = cmap.get(canon_name(xcol))
        yc = cmap.get(canon_name(ycol))
        if xc is None or yc is None:
            raise RuntimeError(
                f"CSV exact columns not found: xcol='{xcol}', ycol='{ycol}'. "
                f"Available columns:\n" + "\n".join(map(str, df.columns))
            )

    x = pd.to_numeric(df[xc], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[yc], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    x, y = ensure_sorted_xy(x, y)
    return x, y, str(xc), str(yc)


def load_xy_exact(path: Path, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_xy_mat_exact(path, xcol=xcol, ycol=ycol)
    return load_xy_csv_exact(path, xcol=xcol, ycol=ycol)


# ============================================================
# Stage A model
# ============================================================

def lorentz(x: np.ndarray, A: float, x0: float, gamma: float) -> np.ndarray:
    gamma = np.maximum(gamma, 1e-12)
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2)


def poly_baseline_centered(x: np.ndarray, coeffs: Sequence[float], x_ref: float) -> np.ndarray:
    dx = x - x_ref
    out = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeffs):
        out = out + c * dx**k
    return out


def pack_theta(theta: np.ndarray, degree: int, n_peaks: int) -> Tuple[np.ndarray, List[PeakSeed]]:
    coeffs = theta[: degree + 1]
    peaks = []
    idx = degree + 1
    for _ in range(n_peaks):
        A, x0, gamma = theta[idx: idx + 3]
        peaks.append(PeakSeed(A=float(A), x0=float(x0), gamma=float(gamma), fwhm=float(2 * gamma)))
        idx += 3
    return coeffs, peaks


def eval_stage_a(x: np.ndarray, theta: np.ndarray, degree: int, n_peaks: int, x_ref: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    coeffs, peaks = pack_theta(theta, degree, n_peaks)
    baseline = poly_baseline_centered(x, coeffs, x_ref=x_ref)
    comps = [lorentz(x, p.A, p.x0, p.gamma) for p in peaks]
    yhat = baseline.copy()
    for c in comps:
        yhat += c
    return yhat, baseline, comps


def build_stage_a_guess(
    x: np.ndarray,
    y: np.ndarray,
    cfg: SafeConfig,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ref = float(np.mean(x))
    dx = x - x_ref

    # baseline from edges
    edge_n = max(8, int(0.16 * len(x)))
    xb = np.concatenate([x[:edge_n], x[-edge_n:]])
    yb = np.concatenate([y[:edge_n], y[-edge_n:]])
    coeffs0 = np.polyfit(xb - x_ref, yb, degree)

    p0 = list(coeffs0)
    lb = [-np.inf] * (degree + 1)
    ub = [ np.inf] * (degree + 1)

    x_span = float(x.max() - x.min())
    y_span = float(max(y.max() - y.min(), 1e-12))

    for p in cfg.primary_peaks:
        amp0 = max(float(p.A), 0.1 * y_span)
        x00 = float(p.x0)
        g0 = max(float(p.gamma), 0.05)

        p0.extend([amp0, x00, g0])

        # hard-lock around safe peaks
        width_pad = max(0.50, 0.55 * p.fwhm)
        lb.extend([0.0, x00 - width_pad, max(0.03, 0.35 * p.gamma)])
        ub.extend([8.0 * y_span, x00 + width_pad, max(5.0 * p.gamma, 0.45)])

    return np.array(p0, float), np.array(lb, float), np.array(ub, float)


def detect_ingest_bug(x: np.ndarray, y: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    corr = abs(corr_safe(x, y))
    px = np.polyfit(x, y, 1)
    y_aff = np.polyval(px, x)
    aff_rmse = rmse(y, y_aff)

    sx = float(np.std(x))
    sy = float(np.std(y))
    ratio = sy / max(sx, 1e-15)

    suspicious = (
        corr > 0.9999
        and 0.1 < ratio < 10.0
        and aff_rmse < 1e-6 * max(sy, 1e-12)
    )

    return suspicious, {
        "abs_corr_xy": float(corr),
        "affine_slope": float(px[0]),
        "affine_intercept": float(px[1]),
        "affine_rmse": float(aff_rmse),
        "std_ratio_y_over_x": float(ratio),
    }


def fit_stage_a(
    x: np.ndarray,
    y: np.ndarray,
    cfg: SafeConfig,
) -> Dict[str, Any]:
    degree = int(cfg.baseline_degree)
    n_peaks = len(cfg.primary_peaks)
    x_ref = float(np.mean(x))

    suspicious_ingest, ingest_diag = detect_ingest_bug(x, y)
    if suspicious_ingest:
        raise RuntimeError(
            "Suspicious ingest detected before Stage A fit: y is almost affine-identical to x."
        )

    theta0, lb, ub = build_stage_a_guess(x, y, cfg, degree=degree)
    sigma = max(cfg.sigma_noise, robust_mad_sigma(np.diff(y)), 1e-8)

    def residuals(theta: np.ndarray) -> np.ndarray:
        yhat, _, _ = eval_stage_a(x, theta, degree, n_peaks, x_ref=x_ref)
        return (yhat - y) / sigma

    res = least_squares(
        residuals,
        theta0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=20000,
        verbose=0,
    )

    yhat, baseline, comps = eval_stage_a(x, res.x, degree, n_peaks, x_ref=x_ref)
    coeffs, peaks_fit = pack_theta(res.x, degree, n_peaks)
    resid = y - yhat

    sigma_noise = robust_mad_sigma(resid)
    proxy = build_rtim_proxy(x, peaks_fit, sigma_noise=sigma_noise)

    proxy_peak_x = float(x[int(np.argmax(proxy["Rcore"]))])
    narrow_centers = [p.x0 for p in peaks_fit]
    peak_alignment_error = float(min(abs(proxy_peak_x - c) for c in narrow_centers))
    peak_alignment_ok = bool(
        peak_alignment_error <= max(0.35, 0.20 * min(p.fwhm for p in peaks_fit))
    )

    residual_corr = corr_safe(resid, proxy["Lcore"])
    transfer_supported = bool(peak_alignment_ok and residual_corr > -0.35)

    rmse_val = rmse(y, yhat)
    aic_val = aic_score(y, yhat, k=len(res.x))
    bic_val = bic_score(y, yhat, k=len(res.x))

    # additional sanity: fitted peaks must stay near safe primary peaks
    center_errors = [
        abs(pf.x0 - ps.x0) for pf, ps in zip(sorted(peaks_fit, key=lambda p: p.x0), sorted(cfg.primary_peaks, key=lambda p: p.x0))
    ]
    centers_ok = bool(all(err <= max(0.60, 0.65 * ps.fwhm) for err, ps in zip(center_errors, sorted(cfg.primary_peaks, key=lambda p: p.x0))))

    if not centers_ok:
        transfer_supported = False

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "theta": res.x,
        "baseline_coeffs": [float(c) for c in coeffs],
        "degree": degree,
        "x_ref": x_ref,
        "fit_y": yhat,
        "baseline_y": baseline,
        "components": comps,
        "residual": resid,
        "peaks": [
            {
                "A": float(p.A),
                "x0": float(p.x0),
                "gamma": float(p.gamma),
                "fwhm": float(p.fwhm),
                "Q": float(p.Q),
            }
            for p in peaks_fit
        ],
        "rmse": float(rmse_val),
        "aic": float(aic_val),
        "bic": float(bic_val),
        "sigma_noise": float(sigma_noise),
        "proxy_peak_x": float(proxy_peak_x),
        "peak_alignment_error": float(peak_alignment_error),
        "peak_alignment_ok": bool(peak_alignment_ok),
        "residual_corr_with_lcore": float(residual_corr),
        "centers_ok": bool(centers_ok),
        "center_errors": [float(v) for v in center_errors],
        "transfer_supported": bool(transfer_supported),
        "proxy": proxy,
        "ingest_diag": ingest_diag,
    }


# ============================================================
# RTIM proxy
# ============================================================

def build_rtim_proxy(x: np.ndarray, peaks_fit: Sequence[PeakSeed], sigma_noise: float) -> Dict[str, np.ndarray]:
    Lcore = np.zeros_like(x, dtype=float)
    for p in peaks_fit:
        Lcore += lorentz(x, p.A, p.x0, p.gamma)

    if np.max(Lcore) > 0:
        Lcore = Lcore / np.max(Lcore)

    gammas = np.array([p.gamma for p in peaks_fit], dtype=float)
    amps = np.array([max(p.A, 1e-12) for p in peaks_fit], dtype=float)
    centers = np.array([p.x0 for p in peaks_fit], dtype=float)
    w = amps / np.sum(amps)

    omega0_ref = float(np.sum(w * centers))
    gamma_ref = float(np.sum(w * gammas))

    alpha = Lcore.copy()
    gamma_tr = alpha / (1.0 + 0.8 * alpha)
    Epair_star = gamma_tr / (gamma_ref + gamma_tr + 1e-12)
    Rcore = Epair_star / max(sigma_noise, 1e-12)

    return {
        "omega0_ref": np.full_like(x, omega0_ref),
        "gamma_ref": np.full_like(x, gamma_ref),
        "Lcore": Lcore,
        "alpha": alpha,
        "gamma_tr": gamma_tr,
        "Epair_star": Epair_star,
        "Rcore": Rcore,
    }


# ============================================================
# Optional Stage B
# ============================================================

def ringdown_model(t: np.ndarray, c: float, A: float, tau: float) -> np.ndarray:
    tau = max(tau, 1e-12)
    return c + A * np.exp(-t / tau)


def fit_ringdown(t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    t, y = ensure_sorted_xy(t, y)
    c0 = float(np.median(y[-max(5, len(y) // 8):]))
    A0 = float(max(y[0] - c0, np.ptp(y), 1e-8))
    tau0 = float(max((t.max() - t.min()) / 4.0, 1e-3))

    p0 = np.array([c0, A0, tau0], float)
    lb = np.array([-np.inf, 0.0, 1e-9], float)
    ub = np.array([ np.inf, np.inf, np.inf], float)

    def residuals(p: np.ndarray) -> np.ndarray:
        return ringdown_model(t, *p) - y

    res = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=max(robust_mad_sigma(y), 1e-9),
        max_nfev=12000,
    )

    yfit = ringdown_model(t, *res.x)
    resid = y - yfit
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": {"c": float(res.x[0]), "A": float(res.x[1]), "tau": float(res.x[2])},
        "fit_y": yfit,
        "residual": resid,
        "rmse": float(rmse(y, yfit)),
        "bic": float(bic_score(y, yfit, 3)),
        "aic": float(aic_score(y, yfit, 3)),
        "resid_sigma": float(robust_mad_sigma(resid)),
    }


def closure_model(t: np.ndarray, T0: float, slope_heat: float, t_switch: float, tau_cool: float) -> np.ndarray:
    tau_cool = max(tau_cool, 1e-12)
    t = np.asarray(t, float)
    out = np.empty_like(t)
    heat_mask = t <= t_switch
    cool_mask = ~heat_mask
    Tpeak = T0 + slope_heat * max(t_switch - t.min(), 0.0)
    out[heat_mask] = T0 + slope_heat * (t[heat_mask] - t.min())
    out[cool_mask] = T0 + (Tpeak - T0) * np.exp(-(t[cool_mask] - t_switch) / tau_cool)
    return out


def fit_closure(t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    t, y = ensure_sorted_xy(t, y)
    T0 = float(np.median(y[:max(5, len(y) // 10)]))
    imax = int(np.argmax(y))
    t_switch0 = float(t[imax])
    slope0 = float((y[imax] - T0) / max(t_switch0 - t.min(), 1e-9))
    tau0 = float(max((t.max() - t_switch0) / 4.0, 1e-3))

    p0 = np.array([T0, max(slope0, 1e-8), t_switch0, tau0], float)
    lb = np.array([-np.inf, 0.0, float(t.min()), 1e-9], float)
    ub = np.array([ np.inf, np.inf, float(t.max()), np.inf], float)

    def residuals(p: np.ndarray) -> np.ndarray:
        return closure_model(t, *p) - y

    res = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=max(robust_mad_sigma(y), 1e-9),
        max_nfev=15000,
    )

    yfit = closure_model(t, *res.x)
    resid = y - yfit
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": {
            "T0": float(res.x[0]),
            "slope_heat": float(res.x[1]),
            "t_switch": float(res.x[2]),
            "tau_cool": float(res.x[3]),
        },
        "fit_y": yfit,
        "residual": resid,
        "rmse": float(rmse(y, yfit)),
        "bic": float(bic_score(y, yfit, 4)),
        "aic": float(aic_score(y, yfit, 4)),
        "resid_sigma": float(robust_mad_sigma(resid)),
    }


# ============================================================
# Stage C residual test
# ============================================================

def fit_extra_channel_residual(
    x: np.ndarray,
    residual: np.ndarray,
    primary_peaks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    y = np.asarray(residual, float)
    y_pos = np.maximum(smooth1d(y, 11), 0.0)

    if np.max(y_pos) <= 0:
        return {
            "success": False,
            "message": "No positive residual structure.",
            "params": {"A": 0.0, "x0": float(x[len(x)//2]), "gamma": 0.2},
            "fit_y": np.zeros_like(x),
            "residual": y.copy(),
            "bic": float(bic_score(y, np.zeros_like(y), 0)),
            "bic_null": float(bic_score(y, np.zeros_like(y), 0)),
            "delta_bic_vs_null": 0.0,
            "independent_of_primary": False,
            "amp_ok": False,
            "extra_supported": False,
            "failure_reasons": ["no_positive_residual_structure"],
        }

    x0_guess = float(x[int(np.argmax(y_pos))])
    A0 = float(np.max(y_pos))
    g0 = 0.5

    p0 = np.array([A0, x0_guess, g0], float)
    lb = np.array([0.0, float(x.min()), 0.03], float)
    ub = np.array([10.0 * max(np.ptp(y), 1e-9), float(x.max()), float((x.max() - x.min()) / 2)], float)

    def residuals(p: np.ndarray) -> np.ndarray:
        A, x0, gamma = p
        return lorentz(x, A, x0, gamma) - y

    res = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=max(robust_mad_sigma(y), 1e-9),
        max_nfev=12000,
    )

    A, x0, gamma = [float(v) for v in res.x]
    yfit = lorentz(x, A, x0, gamma)
    bic_extra = float(bic_score(y, yfit, 3))
    bic_null = float(bic_score(y, np.zeros_like(y), 0))
    delta_bic_vs_null = float(bic_extra - bic_null)
    bic_support_for_extra = float(bic_null - bic_extra)

    primary_centers = np.array([float(p["x0"]) for p in primary_peaks], dtype=float)
    primary_gammas = np.array([max(float(p["gamma"]), 1e-9) for p in primary_peaks], dtype=float)
    d = np.abs(primary_centers - x0)
    j = int(np.argmin(d))
    nearest_dist = float(d[j])

    # extra channel should be meaningfully separate from primary peaks
    independent_of_primary = bool(nearest_dist > max(0.80, 1.25 * primary_gammas[j]))
    amp_ok = bool(A > 2.0 * robust_mad_sigma(y))

    failure_reasons = []
    if bic_support_for_extra <= 10.0:
        failure_reasons.append(
            f"extra_does_not_beat_null_by_bic_margin (support={bic_support_for_extra:.4f}, threshold=10)"
        )
    if not independent_of_primary:
        failure_reasons.append("extra_channel_not_independent_of_primary")
    if not amp_ok:
        failure_reasons.append("extra_channel_amplitude_too_small")

    extra_supported = bool(bic_support_for_extra > 10.0 and independent_of_primary and amp_ok)

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": {"A": A, "x0": x0, "gamma": gamma},
        "fit_y": yfit,
        "residual": y - yfit,
        "rmse": float(rmse(y, yfit)),
        "bic": bic_extra,
        "bic_null": bic_null,
        "delta_bic_vs_null": delta_bic_vs_null,
        "bic_support_for_extra": bic_support_for_extra,
        "nearest_primary_distance": nearest_dist,
        "independent_of_primary": independent_of_primary,
        "amp_ok": amp_ok,
        "extra_supported": extra_supported,
        "failure_reasons": failure_reasons,
    }


# ============================================================
# Verdict
# ============================================================

def final_verdict(
    cfg: SafeConfig,
    stage_a: Dict[str, Any],
    ringdown: Optional[Dict[str, Any]],
    closure: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    notes: List[str] = []

    A_primary_ok = bool(
        cfg.transfer_supported
        and stage_a["success"]
        and stage_a["transfer_supported"]
        and stage_a["centers_ok"]
    )

    if not A_primary_ok:
        notes.append("Stage A failed safe-primary sanity.")
        return {
            "A_primary_ok": False,
            "B_ringdown_ok": None,
            "B_closure_ok": None,
            "B_all_ok": None,
            "C_extra_supported": False if extra is not None else None,
            "transfer_supported_final": False,
            "notes": notes,
            "stage_c_details": extra,
        }

    B_ringdown_ok = None
    if ringdown is not None:
        B_ringdown_ok = bool(
            ringdown["success"]
            and ringdown["params"]["tau"] > 0
            and ringdown["resid_sigma"] < 0.75 * max(np.std(ringdown["fit_y"]), 1e-12)
        )
    else:
        notes.append("Stage B ringdown not tested.")

    B_closure_ok = None
    if closure is not None:
        B_closure_ok = bool(
            closure["success"]
            and closure["params"]["tau_cool"] > 0
            and closure["resid_sigma"] < 0.75 * max(np.std(closure["fit_y"]), 1e-12)
        )
    else:
        notes.append("Stage B closure not tested.")

    if B_ringdown_ok is None and B_closure_ok is None:
        B_all_ok = None
        notes.append("Stage B overall: undetermined (no Stage B data).")
    else:
        flags = [v for v in [B_ringdown_ok, B_closure_ok] if v is not None]
        B_all_ok = bool(all(flags))

    C_extra_supported = None
    if extra is not None:
        C_extra_supported = bool(extra["extra_supported"])
        if not C_extra_supported:
            notes.extend([f"Stage C: {r}" for r in extra["failure_reasons"]])

    transfer_supported_final = None
    if B_all_ok is None:
        transfer_supported_final = None
    else:
        transfer_supported_final = bool(A_primary_ok and B_all_ok and (C_extra_supported if C_extra_supported is not None else True))

    return {
        "A_primary_ok": A_primary_ok,
        "B_ringdown_ok": B_ringdown_ok,
        "B_closure_ok": B_closure_ok,
        "B_all_ok": B_all_ok,
        "C_extra_supported": C_extra_supported,
        "transfer_supported_final": transfer_supported_final,
        "notes": notes,
        "stage_c_details": extra,
    }


# ============================================================
# Plotting / export
# ============================================================

def save_core_table(path: Path, x: np.ndarray, y: np.ndarray, stage_a: Dict[str, Any]) -> None:
    proxy = stage_a["proxy"]
    data = {
        "x": x,
        "y": y,
        "yhat": stage_a["fit_y"],
        "baseline": stage_a["baseline_y"],
        "residual": stage_a["residual"],
        "Lcore": proxy["Lcore"],
        "alpha": proxy["alpha"],
        "gamma_tr": proxy["gamma_tr"],
        "Epair_star": proxy["Epair_star"],
        "Rcore": proxy["Rcore"],
    }
    for i, comp in enumerate(stage_a["components"], start=1):
        data[f"mode_{i}"] = comp
    pd.DataFrame(data).to_csv(path, index=False)


# --- fix ---
def parse_safe_summary(path: str) -> SafeConfig:
    js = json.loads(Path(path).read_text(encoding="utf-8"))

    data_path = js.get("data_path") or js.get("data_source")
    xcol = js.get("xcol") or js.get("exact_xcol_used")
    ycol = js.get("ycol") or js.get("exact_ycol_used")

    fit_window = js.get("fit_window", {})
    fw_min = fit_window.get("xmin")
    fw_max = fit_window.get("xmax")

    cw = None
    if isinstance(js.get("core_fit"), dict):
        cw = js["core_fit"].get("core_window")

    if isinstance(cw, (list, tuple)) and len(cw) == 2:
        core_window = (float(cw[0]), float(cw[1]))
    else:
        cwo = js.get("core_window")
        if isinstance(cwo, dict) and "xmin" in cwo and "xmax" in cwo:
            core_window = (float(cwo["xmin"]), float(cwo["xmax"]))
        elif isinstance(cwo, (list, tuple)) and len(cwo) == 2:
            core_window = (float(cwo[0]), float(cwo[1]))
        elif "x_min" in js and "x_max" in js:
            core_window = (float(js["x_min"]), float(js["x_max"]))
        elif "window" in js and isinstance(js["window"], (list, tuple)) and len(js["window"]) == 2:
            core_window = (float(js["window"][0]), float(js["window"][1]))
        else:
            raise RuntimeError("No usable core window in summary.")

    baseline_degree = int(
        (js.get("core_fit") or {}).get("baseline_degree")
        or js.get("baseline_degree")
        or 1
    )

    sigma_noise = float(
        (js.get("core_fit") or {}).get("sigma_noise")
        or js.get("sigma_noise")
        or js.get("sigma_noise_used")
        or js.get("rmse")
        or 1e-6
    )

    transfer_supported = bool(
        (js.get("core_fit") or {}).get(
            "transfer_supported",
            (js.get("verdict") or {}).get("transfer_supported", js.get("transfer_supported", True))
        )
    )

    seed_peaks_raw = (
        js.get("core_seed_peaks")
        or js.get("core_peaks")
        or js.get("peaks")
        or []
    )

    primary_peaks_raw = (
        (js.get("core_fit") or {}).get("peaks")
        or js.get("core_peaks")
        or js.get("peaks")
        or []
    )

    def _conv_peak(p: dict) -> PeakSeed:
        A = p.get("A", p.get("amp", p.get("amplitude", p.get("height_proxy", 1.0))))
        x0 = p.get("x0", p.get("mu", p.get("center", 0.0)))
        gamma = p.get("gamma", p.get("width", 1.0))
        fwhm = p.get("fwhm", p.get("fwhm_proxy", 2.0 * abs(float(gamma))))
        return PeakSeed(
            A=float(A),
            x0=float(x0),
            gamma=float(abs(gamma)),
            fwhm=float(abs(fwhm)),
        )

    seed_peaks = [_conv_peak(p) for p in seed_peaks_raw if isinstance(p, dict)]
    primary_peaks = [_conv_peak(p) for p in primary_peaks_raw if isinstance(p, dict)]

    if not primary_peaks:
        primary_peaks = seed_peaks

    if not primary_peaks:
        raise RuntimeError("No peaks found in summary.")

    return SafeConfig(
        data_path=data_path,
        xcol=xcol,
        ycol=ycol,
        fit_window=(fw_min, fw_max),
        core_window=core_window,
        baseline_degree=baseline_degree,
        sigma_noise=sigma_noise,
        transfer_supported=transfer_supported,
        seed_peaks=seed_peaks,
        primary_peaks=primary_peaks,
    )


def choose_xy_pair_auto(
    candidates: Sequence[Tuple[str, np.ndarray]],
    dataset_key: Optional[str] = None,
) -> Tuple[str, np.ndarray, str, np.ndarray]:
    pool = list(candidates)

    if dataset_key:
        dk = canon_name(dataset_key)
        filtered = [(n, a) for n, a in pool if dk in canon_name(n)]
        if filtered:
            pool = filtered

    x_tokens = ["freq", "frequency", "omega", "detuning", "energy", "time", ".x", "/x", "x["]
    y_tokens = ["count", "counts", "signal", "intensity", "power", "amplitude", ".y", "/y", "y["]

    best = None
    best_score = None

    for nx, x in pool:
        x = np.asarray(x, float)
        if x.ndim != 1 or len(x) < 8:
            continue
        dx = np.diff(x)
        mono = max(np.mean(dx >= 0), np.mean(dx <= 0))

        for ny, y in pool:
            if nx == ny:
                continue
            y = np.asarray(y, float)
            if y.ndim != 1 or len(y) != len(x):
                continue

            cnx = canon_name(nx)
            cny = canon_name(ny)

            score = (
                int(1000 * mono),
                sum(tok in cnx for tok in x_tokens),
                sum(tok in cny for tok in y_tokens),
                len(x),
            )

            if best_score is None or score > best_score:
                best_score = score
                best = (nx, x, ny, y)

    if best is None:
        raise RuntimeError("No usable auto x/y pair found in MAT candidates.")

    return best


def load_xy_mat_auto(path: Path, dataset_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str, str]:
    candidates: List[Tuple[str, np.ndarray]] = []
    errors = []

    try:
        data = loadmat(path, simplify_cells=True)
        data = {k: matlab_to_python(v) for k, v in data.items() if not str(k).startswith("__")}
        candidates.extend(collect_numeric_candidates(data, prefix=path.stem))
    except Exception as e:
        errors.append(f"loadmat: {e}")

    if not candidates:
        try:
            candidates.extend(collect_hdf5_candidates(path))
        except Exception as e:
            errors.append(f"h5py: {e}")

    if not candidates:
        raise RuntimeError("MAT auto loader found no numeric candidates. " + "; ".join(errors))

    xname, x, yname, y = choose_xy_pair_auto(candidates, dataset_key=dataset_key)
    x, y = ensure_sorted_xy(np.asarray(x, float), np.asarray(y, float))
    return x, y, xname, yname


def load_xy_auto(path: Path, dataset_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str, str]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_xy_mat_auto(path, dataset_key=dataset_key)

    if suffix in {".tsv", ".txt", ".dat"}:
        df = pd.read_csv(path, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    num_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() >= 8:
            num_cols.append(c)

    if len(num_cols) < 2:
        raise RuntimeError("Auto loader found fewer than 2 numeric columns.")

    xc, yc = num_cols[0], num_cols[1]
    x = pd.to_numeric(df[xc], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[yc], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = ensure_sorted_xy(x[m], y[m])
    return x, y, str(xc), str(yc)


def make_plot(
    outpath: Path,
    x: np.ndarray,
    y: np.ndarray,
    stage_a: Dict[str, Any],
    ring_t: Optional[np.ndarray],
    ring_y: Optional[np.ndarray],
    ring_res: Optional[Dict[str, Any]],
    clos_t: Optional[np.ndarray],
    clos_y: Optional[np.ndarray],
    clos_res: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]],
    verdict: Dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9], hspace=0.30, wspace=0.25)

    proxy = stage_a["proxy"]

    # A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, s=10, alpha=0.75, label="real data")
    ax1.plot(x, stage_a["fit_y"], lw=2.3, label="Stage A fit")
    ax1.plot(x, stage_a["baseline_y"], "--", lw=1.8, label="baseline")
    colors = ["tab:red", "tab:purple", "tab:green", "tab:orange"]
    for i, comp in enumerate(stage_a["components"]):
        c = colors[i % len(colors)]
        ax1.plot(x, stage_a["baseline_y"] + comp, ":", lw=1.8, color=c, label=f"mode {i+1}")
        ax1.axvline(stage_a["peaks"][i]["x0"], color=c, alpha=0.15)
    ax1.set_title("A) Real spectral anchor")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Signal")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    # A'
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2.plot(x, proxy["Lcore"], lw=2.0, label="L_core")
    ax2.plot(x, proxy["gamma_tr"], lw=2.0, label="Gamma_tr")
    ax2.plot(x, proxy["Epair_star"], lw=2.0, label="E_pair")
    ax2.plot(x, proxy["Rcore"] / max(np.max(proxy["Rcore"]), 1e-12), lw=2.0, label="R_core (norm)")
    ax2.set_title("A') Anchored RTIM proxy")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Proxy amplitude")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    # B1
    ax3 = fig.add_subplot(gs[1, 0])
    if ring_t is None or ring_y is None:
        ax3.text(0.5, 0.5, "No ringdown data provided", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()
    else:
        ax3.scatter(ring_t, ring_y, s=10, alpha=0.7, label="ringdown data")
        if ring_res is not None:
            ax3.plot(ring_t, ring_res["fit_y"], lw=2.2, label="ringdown fit")
        ax3.set_title("B1) Ringdown")
        ax3.set_xlabel("t")
        ax3.set_ylabel("alpha(t)")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="best")

    # B2
    ax4 = fig.add_subplot(gs[1, 1])
    if clos_t is None or clos_y is None:
        ax4.text(0.5, 0.5, "No closure data provided", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_axis_off()
    else:
        ax4.scatter(clos_t, clos_y, s=10, alpha=0.7, label="closure data")
        if clos_res is not None:
            ax4.plot(clos_t, clos_res["fit_y"], lw=2.2, label="closure fit")
        ax4.set_title("B2) Closure / thermal")
        ax4.set_xlabel("t")
        ax4.set_ylabel("T(t)")
        ax4.grid(alpha=0.25)
        ax4.legend(loc="best")

    # C
    ax5 = fig.add_subplot(gs[2, 0], sharex=ax1)
    resid = stage_a["residual"]
    ax5.scatter(x, resid, s=10, alpha=0.7, label="Stage A residual")
    if extra is not None and extra["success"]:
        ax5.plot(x, extra["fit_y"], lw=2.1, label="extra-channel fit")
        ax5.axvline(extra["params"]["x0"], color="tab:red", ls="--", alpha=0.5)
    ax5.axhline(0.0, color="black", lw=1.0, alpha=0.4)
    ax5.set_title("C) Residual extra-channel test")
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel("Residual")
    ax5.grid(alpha=0.25)
    ax5.legend(loc="best")

    # verdict text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_axis_off()
    lines = [
        "FUZE / RTIM — ABC verdict",
        "",
        f"A_primary_ok         : {verdict['A_primary_ok']}",
        f"B_ringdown_ok        : {verdict['B_ringdown_ok']}",
        f"B_closure_ok         : {verdict['B_closure_ok']}",
        f"B_all_ok             : {verdict['B_all_ok']}",
        f"C_extra_supported    : {verdict['C_extra_supported']}",
        f"transfer_supported   : {verdict['transfer_supported_final']}",
    ]
    if extra is not None:
        lines.append(f"bic_support_for_extra: {extra.get('bic_support_for_extra')}")
    lines.extend(["", "Notes:"])
    for n in verdict["notes"]:
        lines.append(f" - {n}")

    ax6.text(
        0.02, 0.98, "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
        transform=ax6.transAxes,
    )

    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="FUZE / RTIM ABC clean v3")
    ap.add_argument("--core-summary", required=True, help="Path to safe core_summary.json")
    ap.add_argument("--spectrum", default=None, help="Optional override for spectrum path; default = data_path from safe summary")
    ap.add_argument("--ringdown", default=None, help="Optional ringdown CSV/TSV/XLSX")
    ap.add_argument("--ringdown-xcol", default=None)
    ap.add_argument("--ringdown-ycol", default=None)
    ap.add_argument("--closure", default=None, help="Optional closure CSV/TSV/XLSX")
    ap.add_argument("--closure-xcol", default=None)
    ap.add_argument("--closure-ycol", default=None)
    ap.add_argument("--out-prefix", default="fuze_abc_real_v3_clean")
    ap.add_argument("--dataset-key", default=None, help="MAT dataset key for non-exact summaries")
    ap.add_argument("--xcol", default=None, help="Optional exact x column override")
    ap.add_argument("--ycol", default=None, help="Optional exact y column override")

    args = ap.parse_args()

    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        return obj

    cfg = parse_safe_summary(args.core_summary)

    spectrum_path = Path(args.spectrum or cfg.data_path or "")
    if not spectrum_path.exists():
        raise RuntimeError("Spectrum path not found. Pass --spectrum or provide data_path in safe summary.")

    hard_xcol = args.xcol or cfg.xcol
    hard_ycol = args.ycol or cfg.ycol

    print("=== FUZE / RTIM ABC CLEAN v3 ===")
    print(f"Safe summary          : {args.core_summary}")
    print(f"Spectrum path         : {spectrum_path}")
    print(f"Hard xcol             : {hard_xcol}")
    print(f"Hard ycol             : {hard_ycol}")
    print(f"Dataset key           : {args.dataset_key}")

    if hard_xcol and hard_ycol:
        x, y, xc_used, yc_used = load_xy_exact(spectrum_path, hard_xcol, hard_ycol)
        print(f"Loaded exact columns  : {xc_used} | {yc_used}")
    else:
        x, y, xc_used, yc_used = load_xy_auto(spectrum_path, dataset_key=args.dataset_key)
        print(f"Loaded auto columns   : {xc_used} | {yc_used}")

    # hard-lock safe core window
    wmin, wmax = cfg.core_window
    mask = (x >= wmin) & (x <= wmax)
    if mask.sum() < 20:
        raise RuntimeError("Too few points inside safe core window.")
    x_fit = x[mask]
    y_fit = y[mask]

    print(f"Safe core window      : [{x_fit.min():.4f}, {x_fit.max():.4f}]")
    print(f"Seed transfer_support : {cfg.transfer_supported}")

    stage_a = fit_stage_a(x_fit, y_fit, cfg)
    print(f"Stage A RMSE          : {stage_a['rmse']:.6g}")
    print(f"Stage A BIC           : {stage_a['bic']:.6f}")
    print(f"Stage A transfer      : {stage_a['transfer_supported']}")
    print(f"Stage A center errors : {stage_a['center_errors']}")

    ring_t = ring_y = None
    ring_res = None
    if args.ringdown:
        ring_path = Path(args.ringdown)
        if args.ringdown_xcol and args.ringdown_ycol:
            ring_t, ring_y, _, _ = load_xy_csv_exact(ring_path, args.ringdown_xcol, args.ringdown_ycol)
        else:
            ring_t, ring_y, _ = ensure_sorted_xy(*load_xy_csv_exact(ring_path, list(pd.read_csv(ring_path).columns)[0], list(pd.read_csv(ring_path).columns)[1])[:2]), "", ""
            ring_t = ring_t[0]  # dead path avoidance

    # simple cleaner optional loaders for B
    if args.ringdown:
        rdf = pd.read_csv(args.ringdown, sep=None, engine="python")
        if args.ringdown_xcol and args.ringdown_ycol:
            rx, ry = args.ringdown_xcol, args.ringdown_ycol
        else:
            rx, ry = rdf.columns[0], rdf.columns[1]
        ring_t = pd.to_numeric(rdf[rx], errors="coerce").to_numpy(dtype=float)
        ring_y = pd.to_numeric(rdf[ry], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(ring_t) & np.isfinite(ring_y)
        ring_t, ring_y = ensure_sorted_xy(ring_t[m], ring_y[m])
        ring_res = fit_ringdown(ring_t, ring_y)
        print(f"Ringdown RMSE         : {ring_res['rmse']:.6g}")

    clos_t = clos_y = None
    clos_res = None
    if args.closure:
        cdf = pd.read_csv(args.closure, sep=None, engine="python")
        if args.closure_xcol and args.closure_ycol:
            cx, cy = args.closure_xcol, args.closure_ycol
        else:
            cx, cy = cdf.columns[0], cdf.columns[1]
        clos_t = pd.to_numeric(cdf[cx], errors="coerce").to_numpy(dtype=float)
        clos_y = pd.to_numeric(cdf[cy], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(clos_t) & np.isfinite(clos_y)
        clos_t, clos_y = ensure_sorted_xy(clos_t[m], clos_y[m])
        clos_res = fit_closure(clos_t, clos_y)
        print(f"Closure RMSE          : {clos_res['rmse']:.6g}")

    extra = None
    if stage_a["transfer_supported"]:
        extra = fit_extra_channel_residual(x_fit, stage_a["residual"], stage_a["peaks"])
        print(f"Extra BIC support     : {extra['bic_support_for_extra']:.6f}")
        print(f"Extra independent     : {extra['independent_of_primary']}")
        print(f"Extra amp OK          : {extra['amp_ok']}")
    else:
        extra = {
            "success": False,
            "message": "Stage A failed, Stage C skipped.",
            "bic_support_for_extra": None,
            "independent_of_primary": False,
            "amp_ok": False,
            "extra_supported": False,
            "failure_reasons": ["stage_a_failed_so_stage_c_skipped"],
        }

    verdict = final_verdict(cfg, stage_a, ring_res, clos_res, extra)

    out_prefix = Path(args.out_prefix)
    out_summary = Path(f"{out_prefix}_summary.json")
    out_table = Path(f"{out_prefix}_core_table.csv")
    out_plot = Path(f"{out_prefix}_plot.png")

    save_core_table(out_table, x_fit, y_fit, stage_a)
    make_plot(
        out_plot,
        x_fit, y_fit, stage_a,
        ring_t, ring_y, ring_res,
        clos_t, clos_y, clos_res,
        extra, verdict
    )

    summary = {
        "data_source": str(spectrum_path),
        "exact_xcol_used": xc_used,
        "exact_ycol_used": yc_used,
        "window": [float(x_fit.min()), float(x_fit.max())],
        "seed_transfer_supported": bool(cfg.transfer_supported),
        "safe_core_window": [float(cfg.core_window[0]), float(cfg.core_window[1])],
        "stage_A": {
            "success": bool(stage_a["success"]),
            "message": stage_a["message"],
            "rmse": stage_a["rmse"],
            "bic": stage_a["bic"],
            "aic": stage_a["aic"],
            "sigma_noise": stage_a["sigma_noise"],
            "baseline_degree": cfg.baseline_degree,
            "baseline_coeffs": stage_a["baseline_coeffs"],
            "peaks": stage_a["peaks"],
            "proxy_peak_x": stage_a["proxy_peak_x"],
            "peak_alignment_error": stage_a["peak_alignment_error"],
            "peak_alignment_ok": stage_a["peak_alignment_ok"],
            "residual_corr_with_lcore": stage_a["residual_corr_with_lcore"],
            "centers_ok": stage_a["centers_ok"],
            "center_errors": stage_a["center_errors"],
            "transfer_supported": stage_a["transfer_supported"],
            "ingest_diag": stage_a["ingest_diag"],
        },
        "rtim_proxy": {
            "omega0_ref": float(stage_a["proxy"]["omega0_ref"][0]),
            "gamma_ref": float(stage_a["proxy"]["gamma_ref"][0]),
            "Rcore_peak": float(np.max(stage_a["proxy"]["Rcore"])),
            "Rcore_peak_x": float(x_fit[int(np.argmax(stage_a["proxy"]["Rcore"]))]),
            "Epair_peak": float(np.max(stage_a["proxy"]["Epair_star"])),
            "Epair_peak_x": float(x_fit[int(np.argmax(stage_a["proxy"]["Epair_star"]))]),
        },
        "stage_B_ringdown": None if ring_res is None else {
            "success": ring_res["success"],
            "message": ring_res["message"],
            "params": ring_res["params"],
            "rmse": ring_res["rmse"],
            "bic": ring_res["bic"],
            "aic": ring_res["aic"],
            "resid_sigma": ring_res["resid_sigma"],
        },
        "stage_B_closure": None if clos_res is None else {
            "success": clos_res["success"],
            "message": clos_res["message"],
            "params": clos_res["params"],
            "rmse": clos_res["rmse"],
            "bic": clos_res["bic"],
            "aic": clos_res["aic"],
            "resid_sigma": clos_res["resid_sigma"],
        },
        "stage_C_extra": extra,
        "final_verdict": verdict,
    }

    summary_json = to_jsonable(summary)
    out_summary.write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== FINAL VERDICT ===")
    print(f"A_primary_ok         : {verdict['A_primary_ok']}")
    print(f"B_ringdown_ok        : {verdict['B_ringdown_ok']}")
    print(f"B_closure_ok         : {verdict['B_closure_ok']}")
    print(f"B_all_ok             : {verdict['B_all_ok']}")
    print(f"C_extra_supported    : {verdict['C_extra_supported']}")
    print(f"transfer_supported   : {verdict['transfer_supported_final']}")

    print("\nSaved:")
    print(f"  {out_summary}")
    print(f"  {out_table}")
    print(f"  {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())