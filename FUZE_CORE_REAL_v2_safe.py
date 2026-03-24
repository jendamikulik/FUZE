#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
FUZE_CORE_REAL_v2_safe.py
-------------------------

Built directly on the working FUZE_CORE_REAL_v2.py pipeline.

Principle:
- keep the stable Stage-A core extraction exactly as the primary path,
- do not let any extra anchor competition overwrite the primary core fit,
- add only conservative diagnostics and exports.

Outputs:
- core_summary.json
- core_summary.txt
- core_table.csv
- core_real_plot.png

Optional audit:
- a very light alternative 3-peak Lorentz audit can be enabled,
  but it never replaces the primary v2 fit.
"""

import argparse
import json
from dataclasses import dataclass, asdict
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


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class Peak:
    A: float
    x0: float
    gamma: float
    fwhm: float

    @property
    def Q(self) -> float:
        return float(self.x0 / max(self.fwhm, 1e-12))


@dataclass
class CoreFitResult:
    success: bool
    message: str
    rmse: float
    aic: float
    bic: float
    baseline_degree: int
    core_window: Tuple[float, float]
    peaks: List[Peak]
    peak_alignment_error: float
    peak_alignment_ok: bool
    narrow_peak_centers: List[float]
    proxy_peak_x: float
    sigma_noise: float
    residual_corr_with_lcore: float
    proxy_peak_score: float
    transfer_supported: bool


# ----------------------------
# Basic math
# ----------------------------


def lorentz(x: np.ndarray, A: float, x0: float, gamma: float) -> np.ndarray:
    gamma = np.maximum(gamma, 1e-9)
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2)


def poly_baseline(x: np.ndarray, coeffs: Sequence[float], x_ref: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    dx = x - x_ref
    for k, c in enumerate(coeffs):
        y = y + c * dx**k
    return y


def mad_sigma(y: np.ndarray) -> float:
    med = np.median(y)
    return float(1.4826 * np.median(np.abs(y - med)) + 1e-12)


def information_criteria(y: np.ndarray, yhat: np.ndarray, k: int) -> Tuple[float, float, float]:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rmse = float(np.sqrt(rss / max(n, 1)))
    rss_n = max(rss / max(n, 1), 1e-300)
    aic = float(n * np.log(rss_n) + 2 * k)
    bic = float(n * np.log(rss_n) + k * np.log(max(n, 1)))
    return rmse, aic, bic


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ma = np.isfinite(a) & np.isfinite(b)
    if ma.sum() < 5:
        return 0.0
    a = a[ma]
    b = b[ma]
    sa = np.std(a)
    sb = np.std(b)
    if sa <= 1e-15 or sb <= 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------
# Report parsing / core extraction
# ----------------------------


def load_report(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _window_from_report(report: Dict) -> Tuple[float, float]:
    fw = report.get("fit_window", {})
    return float(fw.get("xmin", 0.0)), float(fw.get("xmax", 1.0))


def extract_peaks_from_report(report: Dict) -> List[Peak]:
    peaks = report["best_model_by_BIC"]["peaks"]
    return [Peak(**{k: float(v) for k, v in p.items()}) for p in peaks]


def choose_core_peaks(report: Dict, n_core: int = 2, edge_margin_frac: float = 0.12) -> List[Peak]:
    xmin, xmax = _window_from_report(report)
    span = xmax - xmin
    peaks = extract_peaks_from_report(report)

    scored = []
    for p in peaks:
        edge_dist = min(p.x0 - xmin, xmax - p.x0)
        centrality = edge_dist / max(span, 1e-12)
        edge_penalty = 3.0 if centrality < edge_margin_frac else 0.0
        score = p.gamma + edge_penalty * span
        scored.append((score, p))

    scored.sort(key=lambda t: t[0])
    core = [p for _, p in scored[:n_core]]
    core.sort(key=lambda p: p.x0)
    return core


def core_window_from_peaks(core_peaks: Sequence[Peak], xmin: float, xmax: float, pad_factor: float = 2.0) -> Tuple[float, float]:
    left = min(p.x0 - pad_factor * p.fwhm for p in core_peaks)
    right = max(p.x0 + pad_factor * p.fwhm for p in core_peaks)
    return max(xmin, left), min(xmax, right)


# ----------------------------
# Data loading
# ----------------------------


def _clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = (
                s.astype(str)
                .str.replace(r"\s+", "", regex=True)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^0-9eE+\-\.]+", "", regex=True)
            )
        s = pd.to_numeric(s, errors="coerce")
        if np.mean(np.isfinite(s.to_numpy())) >= 0.7:
            out[str(c)] = s
    if out.shape[1] < 2:
        raise RuntimeError("Po čištění nezůstaly aspoň dva čitelné numerické sloupce.")
    return out


def _guess_xy(df: pd.DataFrame, xcol: Optional[str], ycol: Optional[str]) -> Tuple[str, str]:
    cols = list(df.columns)
    if xcol and ycol:
        return xcol, ycol
    lower = {c: c.lower() for c in cols}
    best_x = None
    best_y = None
    best_x_score = -1
    best_y_score = -1
    for c in cols:
        lc = lower[c]
        vx = pd.to_numeric(df[c], errors="coerce").to_numpy()
        finite = vx[np.isfinite(vx)]
        x_score = int(any(k in lc for k in ["freq", "frequency", "omega", "x"])) + int(np.unique(finite).size > 0.8 * len(finite))
        y_score = int(any(k in lc for k in ["count", "counts", "signal", "amp", "response", "intensity", "y"])) + int(np.nanstd(finite) > 0)
        if x_score > best_x_score:
            best_x_score, best_x = x_score, c
        if y_score > best_y_score:
            best_y_score, best_y = y_score, c
    if best_x is None or best_y is None or best_x == best_y:
        return cols[0], cols[1]
    return best_x, best_y


def _name_score(name: str, kind: str) -> int:
    ln = name.lower()
    if kind == "x":
        keys = ["freq", "frequency", "omega", "ghz", "hz", "energy", "time", "delay", "wavelength", "x"]
        return sum(k in ln for k in keys)
    keys = ["count", "counts", "signal", "amp", "amplitude", "response", "intensity", "power", "trans", "reflection", "y"]
    return sum(k in ln for k in keys)


def _collect_numeric_candidates(obj: Any, prefix: str = "root") -> List[Tuple[str, np.ndarray]]:
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
                if arr.size >= 2:
                    out.append((name, arr))
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
            if arr.size >= 2:
                out.append((name, arr))

    _walk(obj, prefix)
    seen = set()
    uniq: List[Tuple[str, np.ndarray]] = []
    for name, arr in out:
        key = (name, tuple(arr.shape), str(arr.dtype))
        if key not in seen:
            seen.add(key)
            uniq.append((name, arr))
    return uniq


def _collect_hdf5_candidates(path: Path) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    if h5py is None:
        return out
    try:
        with h5py.File(path, "r") as f:
            def visitor(name: str, obj: Any) -> None:
                try:
                    if isinstance(obj, h5py.Dataset):
                        arr = np.array(obj)
                        if arr.dtype.kind in "iufb" and arr.size >= 2:
                            out.append((name, np.squeeze(arr)))
                    elif hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
                        arr = np.array(obj)
                        for fn in arr.dtype.names:
                            sub = np.squeeze(arr[fn])
                            if sub.dtype.kind in "iufb" and sub.size >= 2:
                                out.append((f"{name}.{fn}", sub))
                except Exception:
                    return
            f.visititems(visitor)
    except Exception:
        return []
    return out


def _finalize_xy(x: np.ndarray, y: np.ndarray, xn: str, yn: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3:
        raise RuntimeError("MAT kandidát měl po čištění moc málo bodů.")
    order = np.argsort(x)
    return x[order], y[order], xn, yn


def _choose_xy_from_candidates(cands: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, str, str]:
    best_pair = None
    best_score = -1e18
    for name, arr in cands:
        if arr.ndim != 2 or min(arr.shape) != 2:
            continue
        arr2 = arr if arr.shape[1] == 2 else arr.T
        x = np.asarray(arr2[:, 0], dtype=float)
        y = np.asarray(arr2[:, 1], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 5:
            continue
        dx = np.diff(x[finite])
        monotonic = max(np.mean(dx >= 0), np.mean(dx <= 0))
        uniq_ratio = np.unique(np.round(x[finite], 12)).size / max(finite.sum(), 1)
        score = 10.0 * monotonic + 5.0 * uniq_ratio + _name_score(name, "x") + _name_score(name, "y")
        if score > best_score:
            best_score = score
            best_pair = (x, y, f"{name}[0]", f"{name}[1]")
    if best_pair is not None:
        return _finalize_xy(*best_pair)

    one_d: List[Tuple[str, np.ndarray]] = []
    for name, arr in cands:
        r = np.ravel(arr)
        if r.ndim == 1 and r.size >= 8 and np.isfinite(r).sum() >= 5:
            one_d.append((name, r))
    best = None
    best_score = -1e18
    for i, (nx, ax) in enumerate(one_d):
        for ny, ay in one_d[i + 1 :]:
            n = min(ax.size, ay.size)
            x = np.asarray(ax[:n], dtype=float)
            y = np.asarray(ay[:n], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 5:
                continue
            dx = np.diff(x[finite])
            monotonic = max(np.mean(dx >= 0), np.mean(dx <= 0))
            uniq_ratio = np.unique(np.round(x[finite], 12)).size / max(finite.sum(), 1)
            same_len_bonus = 4.0 if ax.size == ay.size else 0.0
            score_xy = 8.0 * monotonic + 4.0 * uniq_ratio + same_len_bonus + 2.0 * _name_score(nx, "x") + _name_score(ny, "y")
            score_yx = 8.0 * max(np.mean(np.diff(y[finite]) >= 0), np.mean(np.diff(y[finite]) <= 0)) + 4.0 * (np.unique(np.round(y[finite], 12)).size / max(finite.sum(), 1)) + same_len_bonus + 2.0 * _name_score(ny, "x") + _name_score(nx, "y")
            if score_xy > best_score:
                best_score = score_xy
                best = (x, y, nx, ny)
            if score_yx > best_score:
                best_score = score_yx
                best = (y, x, ny, nx)
    if best is not None:
        return _finalize_xy(*best)

    raise RuntimeError("MAT soubor jsem neuměl rozumně rozbalit na x/y.")


def load_xy(path: Path, xcol: Optional[str] = None, ycol: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".tsv", ".txt", ".dat"}:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".mat":
        candidates: List[Tuple[str, np.ndarray]] = []
        errors: List[str] = []
        try:
            data = loadmat(path, simplify_cells=True)
            candidates.extend(_collect_numeric_candidates(data, prefix=path.stem))
        except Exception as e:
            errors.append(f"loadmat: {e}")
        if not candidates:
            try:
                candidates.extend(_collect_hdf5_candidates(path))
            except Exception as e:
                errors.append(f"h5py: {e}")
        if not candidates:
            detail = "; ".join(errors) if errors else "bez detailu"
            raise RuntimeError(f"MAT soubor jsem neuměl rozumně rozbalit na x/y. Detaily: {detail}")
        return _choose_xy_from_candidates(candidates)
    else:
        raise RuntimeError(f"Nepodporovaný formát: {path.suffix}")

    df = _clean_numeric_df(df)
    xc, yc = _guess_xy(df, xcol, ycol)
    x = pd.to_numeric(df[xc], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[yc], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    order = np.argsort(x)
    return x[order], y[order], xc, yc


# ----------------------------
# Core fit
# ----------------------------


def pack_core_params(theta: np.ndarray, degree: int, n_peaks: int) -> Tuple[np.ndarray, List[Peak]]:
    coeffs = theta[: degree + 1]
    peaks = []
    idx = degree + 1
    for _ in range(n_peaks):
        A, x0, gamma = theta[idx: idx + 3]
        peaks.append(Peak(A=float(A), x0=float(x0), gamma=float(gamma), fwhm=float(2 * gamma)))
        idx += 3
    return coeffs, peaks


def core_model(x: np.ndarray, theta: np.ndarray, degree: int, n_peaks: int, x_ref: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    coeffs, peaks = pack_core_params(theta, degree=degree, n_peaks=n_peaks)
    baseline = poly_baseline(x, coeffs, x_ref=x_ref)
    comps = [lorentz(x, p.A, p.x0, p.gamma) for p in peaks]
    y = baseline.copy()
    for c in comps:
        y += c
    return y, baseline, comps


def fit_core_window(
    x: np.ndarray,
    y: np.ndarray,
    core_peaks_seed: Sequence[Peak],
    degree: int = 1,
    x_window: Optional[Tuple[float, float]] = None,
    max_nfev: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_window is not None:
        xmin, xmax = x_window
        m = (x >= xmin) & (x <= xmax)
        x = x[m]
        y = y[m]
    if len(x) < 20:
        raise RuntimeError("V core window je příliš málo bodů pro fit.")

    x_ref = float(np.mean(x))
    idx_low = np.argsort(y)[: max(10, len(y) // 5)]
    coeffs0 = np.polyfit(x[idx_low] - x_ref, y[idx_low], deg=degree)[::-1]
    theta0 = list(coeffs0)
    lb = [-np.inf] * (degree + 1)
    ub = [np.inf] * (degree + 1)

    span = x.max() - x.min()
    yr = max(np.ptp(y), 1e-12)
    for p in core_peaks_seed:
        theta0.extend([p.A, p.x0, p.gamma])
        lb.extend([0.0, max(x.min(), p.x0 - 0.25 * span), 0.05])
        ub.extend([5.0 * yr, min(x.max(), p.x0 + 0.25 * span), 0.5 * span])

    theta0 = np.array(theta0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    sigma = mad_sigma(np.diff(y))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(np.std(y) * 0.05, 1e-6)

    def residuals(theta: np.ndarray) -> np.ndarray:
        yhat, _, _ = core_model(x, theta, degree=degree, n_peaks=len(core_peaks_seed), x_ref=x_ref)
        return (yhat - y) / sigma

    res = least_squares(
        residuals,
        theta0,
        bounds=(lb, ub),
        max_nfev=max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=0,
    )
    return x, y, res.x


# ----------------------------
# RTIM proxy
# ----------------------------


def build_rtim_proxy(x: np.ndarray, peaks: Sequence[Peak], sigma_noise: float, c_s: float = 1.0, Gmax: float = 1.0, alpha_sat: float = 0.35, Pin: float = 1.0, Gloss: float = 0.20, Grel: float = 0.30) -> Dict[str, np.ndarray]:
    Ls = [lorentz(x, 1.0, p.x0, p.gamma) for p in peaks]
    weights = np.array([max(p.A, 1e-12) for p in peaks], dtype=float)
    weights = weights / weights.sum()
    Lcore = np.zeros_like(x)
    for w, Li in zip(weights, Ls):
        Lcore += w * Li

    alpha = c_s * Lcore
    gamma_tr = Gmax * alpha / (alpha + alpha_sat + 1e-12)
    Epair_star = (gamma_tr / max(Grel, 1e-12)) * (Pin / (Gloss + gamma_tr + 1e-12))
    Rcore = Epair_star / max(sigma_noise, 1e-12)

    return {
        "Lcore": Lcore,
        "alpha": alpha,
        "gamma_tr": gamma_tr,
        "Epair_star": Epair_star,
        "Rcore": Rcore,
    }


# ----------------------------
# Optional audit (non-destructive)
# ----------------------------


def fit_three_peak_lorentz_audit(x: np.ndarray, y: np.ndarray, degree: int, seed_peaks: Sequence[Peak], max_nfev: int = 5000) -> Optional[Dict[str, Any]]:
    """
    Very light audit only.
    Starts from the 2 core peaks and adds one broad middle shoulder.
    Never used as the primary anchor.
    """
    if len(x) < 30:
        return None
    x_ref = float(np.mean(x))
    span = float(x.max() - x.min())
    idx_low = np.argsort(y)[: max(10, len(y) // 5)]
    coeffs0 = np.polyfit(x[idx_low] - x_ref, y[idx_low], deg=degree)[::-1]
    theta0 = list(coeffs0)
    lb = [-np.inf] * (degree + 1)
    ub = [np.inf] * (degree + 1)
    yr = max(float(np.ptp(y)), 1e-12)

    # two seed peaks
    for p in seed_peaks:
        theta0.extend([p.A, p.x0, p.gamma])
        lb.extend([0.0, p.x0 - 0.25 * span, 0.05])
        ub.extend([5.0 * yr, p.x0 + 0.25 * span, 0.5 * span])

    # broad central shoulder
    theta0.extend([0.35 * yr, float(np.mean([p.x0 for p in seed_peaks])), 0.18 * span])
    lb.extend([0.0, x.min(), 0.05])
    ub.extend([5.0 * yr, x.max(), 0.75 * span])

    theta0 = np.array(theta0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    sigma = mad_sigma(np.diff(y))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(np.std(y) * 0.05, 1e-6)

    def residuals(theta: np.ndarray) -> np.ndarray:
        yhat, _, _ = core_model(x, theta, degree=degree, n_peaks=3, x_ref=x_ref)
        return (yhat - y) / sigma

    try:
        res = least_squares(
            residuals,
            theta0,
            bounds=(lb, ub),
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            verbose=0,
        )
        yhat, baseline, comps = core_model(x, res.x, degree=degree, n_peaks=3, x_ref=x_ref)
        coeffs, peaks = pack_core_params(res.x, degree=degree, n_peaks=3)
        rmse, aic, bic = information_criteria(y, yhat, k=len(res.x))
        return {
            "success": True,
            "baseline_coeffs": [float(c) for c in coeffs],
            "peaks": [asdict(p) | {"Q": p.Q} for p in peaks],
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Exports / plotting
# ----------------------------


def save_core_table(path: Path, x: np.ndarray, y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray, comps: Sequence[np.ndarray], proxy: Dict[str, np.ndarray]) -> None:
    data = {
        "x": x,
        "y": y,
        "yhat": yhat,
        "baseline": baseline,
        "residual": y - yhat,
        "Lcore": proxy["Lcore"],
        "alpha": proxy["alpha"],
        "gamma_tr": proxy["gamma_tr"],
        "Epair_star": proxy["Epair_star"],
        "Rcore": proxy["Rcore"],
    }
    for i, c in enumerate(comps, start=1):
        data[f"mode_{i}"] = c
    pd.DataFrame(data).to_csv(path, index=False)


def save_text_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("FUZE_CORE_REAL_v2_safe report")
    lines.append("============================")
    lines.append("")
    lines.append(f"report: {summary.get('report_path')}")
    lines.append(f"mode: {summary.get('mode')}")
    lines.append(f"fit window: [{summary['fit_window']['xmin']:.6f}, {summary['fit_window']['xmax']:.6f}]")
    lines.append(f"core window: [{summary['core_window']['xmin']:.6f}, {summary['core_window']['xmax']:.6f}]")
    lines.append("")
    lines.append("Seed peaks")
    lines.append("----------")
    for p in summary.get("core_seed_peaks", []):
        lines.append(f"x0={p['x0']:.6f}, gamma={p['gamma']:.6f}, fwhm={p['fwhm']:.6f}, Q={p['Q']:.6f}")

    cf = summary.get("core_fit")
    if cf:
        lines.append("")
        lines.append("Primary core fit")
        lines.append("----------------")
        lines.append(f"rmse={cf['rmse']:.6e}")
        lines.append(f"aic={cf['aic']:.6f}")
        lines.append(f"bic={cf['bic']:.6f}")
        lines.append(f"peak_alignment_error={cf['peak_alignment_error']:.6f}")
        lines.append(f"peak_alignment_ok={cf['peak_alignment_ok']}")
        lines.append(f"residual_corr_with_lcore={cf['residual_corr_with_lcore']:.6f}")
        lines.append(f"proxy_peak_score={cf['proxy_peak_score']:.6f}")
        lines.append(f"TRANSFER SUPPORTED? {'YES' if cf['transfer_supported'] else 'NO'}")
        lines.append("")
        for i, p in enumerate(cf["peaks"], start=1):
            lines.append(f"peak {i}: x0={p['x0']:.6f}, gamma={p['gamma']:.6f}, fwhm={p['fwhm']:.6f}, Q={p['Q']:.6f}")

    audit = summary.get("audit")
    if audit:
        lines.append("")
        lines.append("Audit only")
        lines.append("----------")
        if audit.get("success"):
            lines.append(f"3-peak Lorentz audit bic={audit['bic']:.6f}, rmse={audit['rmse']:.6e}")
        else:
            lines.append(f"audit failed: {audit.get('error', 'unknown error')}")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_plot(
    x: np.ndarray,
    y: Optional[np.ndarray],
    yhat: Optional[np.ndarray],
    baseline: Optional[np.ndarray],
    comps: Optional[Sequence[np.ndarray]],
    proxy_x: np.ndarray,
    proxy: Dict[str, np.ndarray],
    core_peaks: Sequence[Peak],
    outpath: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.2, 0.9], hspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    if y is not None:
        ax1.plot(x, y, ".", ms=4, alpha=0.65, label="real data")
    if yhat is not None:
        ax1.plot(x, yhat, "-", lw=2.2, label="primary core fit")
    if baseline is not None:
        ax1.plot(x, baseline, "--", lw=1.8, label="baseline")
    if comps is not None:
        for i, c in enumerate(comps, start=1):
            ax1.plot(x, c + (baseline if baseline is not None else 0), ":", lw=1.5, label=f"mode {i}")
    for p in core_peaks:
        ax1.axvline(p.x0, color="k", lw=1.0, alpha=0.2)
    ax1.set_title(title)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Signal")
    ax1.legend(fontsize=9, ncol=2)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(proxy_x, proxy["Lcore"], label="L_core")
    ax2.plot(proxy_x, proxy["gamma_tr"], label="Gamma_tr")
    ax2.plot(proxy_x, proxy["Epair_star"], label="E_pair* proxy")
    for p in core_peaks:
        ax2.axvline(p.x0, color="k", lw=1.0, alpha=0.2)
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Proxy amplitude")
    ax2.set_title("RTIM core proxy anchored to the real resonances")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(proxy_x, proxy["Rcore"], label="R_core = E_pair*/sigma", lw=2)
    peak_x = proxy_x[int(np.argmax(proxy["Rcore"]))]
    ax3.axvline(peak_x, color="r", ls="--", lw=1.2, label=f"proxy peak = {peak_x:.3f}")
    for p in core_peaks:
        ax3.axvline(p.x0, color="k", lw=1.0, alpha=0.25)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Detectability")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, Dict
from pathlib import Path


def make_plot(
        x: np.ndarray,
        y: Optional[np.ndarray],
        yhat: Optional[np.ndarray],
        baseline: Optional[np.ndarray],
        comps: Optional[Sequence[np.ndarray]],
        proxy_x: np.ndarray,
        proxy: Dict[str, np.ndarray],
        core_peaks: Sequence,  # Peak type
        outpath: Path,
        title: str,
) -> None:
    # 1. Definujeme layout="constrained" přímo při vytváření figury
    fig = plt.figure(figsize=(13, 9), layout="constrained")

    # 2. Gridspec nyní necháme bez hspace, o to se postará constrained layout
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.2, 0.9])

    # --- Horní graf (Data a Fit) ---
    ax1 = fig.add_subplot(gs[0])
    if y is not None:
        ax1.plot(x, y, ".", ms=4, alpha=0.65, label="real data")
    if yhat is not None:
        ax1.plot(x, yhat, "-", lw=2.2, label="primary core fit")
    if baseline is not None:
        ax1.plot(x, baseline, "--", lw=1.8, label="baseline")
    if comps is not None:
        for i, c in enumerate(comps, start=1):
            ax1.plot(x, c + (baseline if baseline is not None else 0), ":", lw=1.5, label=f"mode {i}")
    for p in core_peaks:
        ax1.axvline(p.x0, color="k", lw=1.0, alpha=0.2)
    ax1.set_title(title)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Signal")
    ax1.legend(fontsize=9, ncol=2)

    # --- Prostřední graf (Proxy parametry) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(proxy_x, proxy["Lcore"], label="L_core")
    ax2.plot(proxy_x, proxy["gamma_tr"], label="Gamma_tr")
    ax2.plot(proxy_x, proxy["Epair_star"], label="E_pair* proxy")
    for p in core_peaks:
        ax2.axvline(p.x0, color="k", lw=1.0, alpha=0.2)
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Proxy amplitude")
    ax2.set_title("RTIM core proxy anchored to the real resonances")
    ax2.legend(fontsize=9)

    # --- Spodní graf (Detekovatelnost) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(proxy_x, proxy["Rcore"], label="R_core = E_pair*/sigma", lw=2)
    peak_x = proxy_x[int(np.argmax(proxy["Rcore"]))]
    ax3.axvline(peak_x, color="r", ls="--", lw=1.2, label=f"proxy peak = {peak_x:.3f}")
    for p in core_peaks:
        ax3.axvline(p.x0, color="k", lw=1.0, alpha=0.25)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Detectability")
    ax3.legend(fontsize=9)

    # 3. fig.tight_layout() JIŽ NENÍ POTŘEBA (vyřešeno v kroku 1)

    # 4. Ukládáme s bbox_inches='tight' pro jistotu absolutně čistých okrajů
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Main runner
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Safe Stage-A core RTIM runner over real spectral data")
    ap.add_argument("--report", default="fit_report.json", help="path to fit_report.json")
    ap.add_argument("--csv", default="fig2c_data.mat", help="path to clean_spectrum.csv / tsv / xlsx / mat")
    ap.add_argument("--xcol", default=None)
    ap.add_argument("--ycol", default=None)
    ap.add_argument("--degree", type=int, default=1, choices=[0, 1, 2], help="baseline degree for core refit")
    ap.add_argument("--outdir", default="core_out")
    ap.add_argument("--max-nfev", type=int, default=4000)
    ap.add_argument("--audit-competition", action="store_true", default=True, help="run a light 3-peak Lorentz audit without altering the primary fit")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = load_report(Path(args.report))
    xmin, xmax = _window_from_report(report)
    core_peaks_seed = choose_core_peaks(report, n_core=2)
    core_window = core_window_from_peaks(core_peaks_seed, xmin=xmin, xmax=xmax, pad_factor=2.0)

    summary: Dict[str, Any] = {
        "report_path": str(Path(args.report).resolve()),
        "fit_window": {"xmin": xmin, "xmax": xmax},
        "best_model_BIC": report["best_model_by_BIC"]["BIC"],
        "best_model_n_peaks": report["best_model_by_BIC"]["n_peaks"],
        "core_seed_peaks": [asdict(p) | {"Q": p.Q} for p in core_peaks_seed],
        "core_window": {"xmin": core_window[0], "xmax": core_window[1]},
        "notes": [
            "Primary path is the original working v2 Stage-A fit.",
            "Any competition/audit is optional and never replaces the primary anchor.",
        ],
    }

    if args.csv:
        x, y, xc, yc = load_xy(Path(args.csv), xcol=args.xcol, ycol=args.ycol)
        xfit, yfit, theta = fit_core_window(x, y, core_peaks_seed, degree=args.degree, x_window=core_window, max_nfev=args.max_nfev)
        x_ref = float(np.mean(xfit))
        yhat, baseline, comps = core_model(xfit, theta, degree=args.degree, n_peaks=len(core_peaks_seed), x_ref=x_ref)
        coeffs, peaks_fit = pack_core_params(theta, degree=args.degree, n_peaks=len(core_peaks_seed))

        residual = yfit - yhat
        sigma_noise = mad_sigma(residual)
        proxy = build_rtim_proxy(xfit, peaks_fit, sigma_noise=sigma_noise)
        proxy_peak_x = float(xfit[int(np.argmax(proxy["Rcore"]))])
        narrow_centers = [p.x0 for p in peaks_fit]
        peak_alignment_error = float(min(abs(proxy_peak_x - c) for c in narrow_centers))
        peak_alignment_ok = bool(peak_alignment_error <= max(0.35, 0.20 * min(p.fwhm for p in peaks_fit)))
        rmse, aic, bic = information_criteria(yfit, yhat, k=len(theta))
        residual_corr = corr_safe(residual, proxy["Lcore"])
        proxy_peak_score = float(np.max(proxy["Rcore"]))
        transfer_supported = bool(peak_alignment_ok and residual_corr > -0.35)

        result = CoreFitResult(
            success=True,
            message="primary core refit completed",
            rmse=rmse,
            aic=aic,
            bic=bic,
            baseline_degree=args.degree,
            core_window=(float(xfit.min()), float(xfit.max())),
            peaks=peaks_fit,
            peak_alignment_error=peak_alignment_error,
            peak_alignment_ok=peak_alignment_ok,
            narrow_peak_centers=narrow_centers,
            proxy_peak_x=proxy_peak_x,
            sigma_noise=sigma_noise,
            residual_corr_with_lcore=residual_corr,
            proxy_peak_score=proxy_peak_score,
            transfer_supported=transfer_supported,
        )

        summary.update(
            {
                "mode": "report+data",
                "data_path": str(Path(args.csv).resolve()),
                "xcol": xc,
                "ycol": yc,
                "baseline_coeffs": [float(c) for c in coeffs],
                "core_fit": {
                    **asdict(result),
                    "peaks": [asdict(p) | {"Q": p.Q} for p in result.peaks],
                },
                "verdict": {
                    "transfer_supported": transfer_supported,
                    "logic": "primary_v2_stage_a_only",
                    "peak_alignment_ok": peak_alignment_ok,
                    "residual_corr_with_lcore": residual_corr,
                },
            }
        )

        save_core_table(outdir / "core_table.csv", xfit, yfit, yhat, baseline, comps, proxy)

        if args.audit_competition:
            audit = fit_three_peak_lorentz_audit(xfit, yfit, degree=args.degree, seed_peaks=peaks_fit, max_nfev=args.max_nfev)
            summary["audit"] = audit

        make_plot(
            x=xfit,
            y=yfit,
            yhat=yhat,
            baseline=baseline,
            comps=comps,
            proxy_x=xfit,
            proxy=proxy,
            core_peaks=peaks_fit,
            outpath=outdir / "core_real_plot.png",
            title="FUZE / RTIM core fit over real data (safe primary path)",
        )
    else:
        xgrid = np.linspace(core_window[0], core_window[1], 800)
        sigma_noise = 1.0
        proxy = build_rtim_proxy(xgrid, core_peaks_seed, sigma_noise=sigma_noise)
        proxy_peak_x = float(xgrid[int(np.argmax(proxy["Rcore"]))])
        narrow_centers = [p.x0 for p in core_peaks_seed]
        peak_alignment_error = float(min(abs(proxy_peak_x - c) for c in narrow_centers))
        peak_alignment_ok = bool(peak_alignment_error <= max(0.35, 0.20 * min(p.fwhm for p in core_peaks_seed)))

        summary.update(
            {
                "mode": "report-only",
                "report_only_note": "Pass --csv clean_spectrum.csv to do the real-data refit.",
                "proxy_peak_x": proxy_peak_x,
                "peak_alignment_error": peak_alignment_error,
                "peak_alignment_ok": peak_alignment_ok,
                "narrow_peak_centers": narrow_centers,
            }
        )
        make_plot(
            x=xgrid,
            y=None,
            yhat=None,
            baseline=None,
            comps=None,
            proxy_x=xgrid,
            proxy=proxy,
            core_peaks=core_peaks_seed,
            outpath=outdir / "core_report_only_plot.png",
            title="FUZE / RTIM report-only core proxy",
        )

    (outdir / "core_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_text_report(outdir / "core_summary.txt", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
