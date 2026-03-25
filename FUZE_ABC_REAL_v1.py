# FUZE_ABC_REAL_v1.py
# -*- coding: utf-8 -*-

"""
FUZE_ABC_REAL_v1.py

Safe primary-path script:
1) load real spectral data
2) load core_summary.json from the safe Stage-A extractor
3) freeze anchor seeds from real core peaks
4) refit only a stable baseline + Lorentz core model
5) build anchored RTIM proxy over the real core
6) export plot + summary + core table

Usage examples
--------------
python FUZE_ABC_REAL_v1.py ^
  --core-summary core_summary.json ^
  --spectrum fig2c_data.mat ^
  --dataset-key "Sample A" ^
  --out-prefix fuze_abc_real_v1

python FUZE_ABC_REAL_v1.py ^
  --core-summary core_summary.json ^
  --spectrum spectrum.csv ^
  --x-col 0 --y-col 1 ^
  --window-min 27.9 --window-max 52.55 ^
  --out-prefix fuze_abc_real_v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
    from scipy.io import loadmat
    from scipy.io.matlab import mat_struct
except Exception as e:
    raise RuntimeError(
        "This script needs scipy installed. Try: pip install scipy"
    ) from e


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def robust_mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def bic_score(y_true: np.ndarray, y_fit: np.ndarray, k: int) -> float:
    n = len(y_true)
    rss = float(np.sum((y_true - y_fit) ** 2))
    return n * np.log(rss / max(n, 1) + 1e-30) + k * np.log(max(n, 1))


def aic_score(y_true: np.ndarray, y_fit: np.ndarray, k: int) -> float:
    n = len(y_true)
    rss = float(np.sum((y_true - y_fit) ** 2))
    return n * np.log(rss / max(n, 1) + 1e-30) + 2 * k


def rmse(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_fit) ** 2)))


def smooth1d(y: np.ndarray, window: int = 9) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def ensure_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    return x[idx], y[idx]


# ---------------------------------------------------------------------
# MATLAB recursive loader
# ---------------------------------------------------------------------

def matlab_to_python(obj: Any) -> Any:
    if isinstance(obj, mat_struct):
        out = {}
        for f in obj._fieldnames:
            out[f] = matlab_to_python(getattr(obj, f))
        return out
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [matlab_to_python(v) for v in obj.flat]
        return obj
    return obj


def _is_numeric_1d(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) and x.ndim == 1 and x.size >= 8


def _try_curve_from_dict(d: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    keymap = {str(k).lower(): k for k in d.keys()}

    x_keys = ["x", "freq", "frequency", "omega", "xdata", "f"]
    y_keys = ["y", "signal", "counts", "amplitude", "amp", "z", "zss", "z_ss", "ydata"]

    x = None
    y = None

    for k in x_keys:
        if k in keymap and _is_numeric_1d(d[keymap[k]]):
            x = np.asarray(d[keymap[k]], dtype=float).ravel()
            break

    for k in y_keys:
        if k in keymap and _is_numeric_1d(d[keymap[k]]):
            y = np.asarray(d[keymap[k]], dtype=float).ravel()
            break

    if x is not None and y is not None and len(x) == len(y):
        return x, y
    return None


def _try_curve_from_array(arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        return None

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            x = np.asarray(arr[:, 0], dtype=float).ravel()
            y = np.asarray(arr[:, 1], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y
        if arr.shape[0] >= 2:
            x = np.asarray(arr[0, :], dtype=float).ravel()
            y = np.asarray(arr[1, :], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y
    return None


def extract_curves(node: Any, prefix: str = "root") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if isinstance(node, dict):
        maybe = _try_curve_from_dict(node)
        if maybe is not None:
            curves[prefix] = maybe
        for k, v in node.items():
            curves.update(extract_curves(v, f"{prefix}/{k}"))
        return curves

    if isinstance(node, list):
        for i, v in enumerate(node):
            curves.update(extract_curves(v, f"{prefix}[{i}]"))
        return curves

    if isinstance(node, np.ndarray):
        maybe = _try_curve_from_array(node)
        if maybe is not None:
            curves[prefix] = maybe
        if node.dtype == object:
            for i, v in enumerate(node.flat):
                curves.update(extract_curves(v, f"{prefix}[{i}]"))
        return curves

    return curves


def load_spectrum_from_mat(path: str, dataset_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    raw = {k: matlab_to_python(v) for k, v in raw.items() if not k.startswith("__")}
    curves = extract_curves(raw, prefix=os.path.basename(path))

    if not curves:
        raise RuntimeError(f"No usable curve was found in MAT file: {path}")

    names = list(curves.keys())

    if dataset_key:
        key_low = dataset_key.lower()
        matches = [n for n in names if key_low in n.lower()]
        if not matches:
            raise RuntimeError(
                f"Dataset key '{dataset_key}' not found. Available curve names:\n"
                + "\n".join(names[:50])
            )
        chosen = matches[0]
    else:
        chosen = max(names, key=lambda n: len(curves[n][0]))

    x, y = curves[chosen]
    x, y = ensure_sorted_xy(np.asarray(x, float), np.asarray(y, float))
    return x, y, chosen


def load_spectrum_from_csv(path: str, x_col: int = 0, y_col: int = 1) -> Tuple[np.ndarray, np.ndarray, str]:
    df = pd.read_csv(path, sep=None, engine="python")
    x = pd.to_numeric(df.iloc[:, x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    x, y = ensure_sorted_xy(x.astype(float), y.astype(float))
    return x, y, os.path.basename(path)


def load_spectrum(path: str, dataset_key: Optional[str], x_col: int, y_col: int) -> Tuple[np.ndarray, np.ndarray, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return load_spectrum_from_mat(path, dataset_key=dataset_key)
    return load_spectrum_from_csv(path, x_col=x_col, y_col=y_col)


# ---------------------------------------------------------------------
# Core summary parser
# ---------------------------------------------------------------------

@dataclass
class PeakSeed:
    x0: float
    gamma: float
    amp: float


@dataclass
class CoreSeedBundle:
    window_min: Optional[float]
    window_max: Optional[float]
    sigma_noise: float
    transfer_supported: bool
    peaks: List[PeakSeed]


def parse_core_summary(path: str) -> CoreSeedBundle:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)

    window = js.get("core_window", None)
    if isinstance(window, (list, tuple)) and len(window) == 2:
        wmin, wmax = float(window[0]), float(window[1])
    else:
        wmin, wmax = None, None

    sigma_noise = (
        js.get("sigma_noise")
        or js.get("noise_sigma")
        or js.get("residual_sigma")
        or 1e-6
    )
    sigma_noise = float(sigma_noise)

    transfer_supported = bool(js.get("transfer_supported", True))

    peaks_raw = []
    if isinstance(js.get("core_fit"), dict):
        peaks_raw = js["core_fit"].get("peaks", [])
    if not peaks_raw:
        peaks_raw = js.get("peaks", [])

    peaks: List[PeakSeed] = []
    for p in peaks_raw:
        if not isinstance(p, dict):
            continue

        """x0 = float(p.get("x0", p.get("mu", 0.0)))
        gamma = float(abs(p.get("gamma", p.get("width", 1.0))))
        amp = float(max(p.get("amp", p.get("height", 1.0)), 1e-12))"""
        x0 = float(p.get("x0", p.get("mu", p.get("center", 0.0))))
        gamma = float(abs(p.get("gamma", p.get("width", 1.0))))
        amp = float(max(
            p.get("amp", p.get("amplitude", p.get("height", p.get("height_proxy", 1.0)))),
            1e-12
        ))
        
        peaks.append(PeakSeed(x0=x0, gamma=gamma, amp=amp))

    if not peaks:
        raise RuntimeError(
            "No peaks found in core_summary.json. I expected core_fit.peaks or peaks."
        )

    return CoreSeedBundle(
        window_min=wmin,
        window_max=wmax,
        sigma_noise=sigma_noise,
        transfer_supported=transfer_supported,
        peaks=peaks,
    )


# ---------------------------------------------------------------------
# Stage-A stable model: polynomial baseline + Lorentz peaks
# ---------------------------------------------------------------------

def lorentz(x: np.ndarray, amp: float, x0: float, gamma: float) -> np.ndarray:
    g2 = gamma * gamma
    return amp * g2 / ((x - x0) ** 2 + g2)


def eval_stage_a_model(
    x: np.ndarray,
    params: np.ndarray,
    baseline_deg: int,
    n_peaks: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    coeffs = params[: baseline_deg + 1]
    baseline = np.polyval(coeffs, x)
    y = baseline.copy()

    comps: List[np.ndarray] = []
    idx = baseline_deg + 1
    for _ in range(n_peaks):
        amp, x0, gamma = params[idx : idx + 3]
        comp = lorentz(x, amp, x0, gamma)
        y = y + comp
        comps.append(comp)
        idx += 3

    return y, baseline, comps


def build_initial_stage_a_guess(
    x: np.ndarray,
    y: np.ndarray,
    core: CoreSeedBundle,
    baseline_deg: int = 2,
    max_peaks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PeakSeed]]:
    n = len(x)
    edge_n = max(8, int(0.15 * n))
    xb = np.concatenate([x[:edge_n], x[-edge_n:]])
    yb = np.concatenate([y[:edge_n], y[-edge_n:]])
    coeffs0 = np.polyfit(xb, yb, baseline_deg)

    seeds = sorted(core.peaks, key=lambda p: p.x0)
    if max_peaks is not None:
        seeds = seeds[:max_peaks]

    if not seeds:
        raise RuntimeError("No peak seeds available.")

    span = float(x.max() - x.min())
    yspan = float(max(y.max() - y.min(), 1e-8))

    p0 = list(coeffs0)
    lo = [-np.inf] * len(coeffs0)
    hi = [np.inf] * len(coeffs0)

    for s in seeds:
        baseline_at = float(np.polyval(coeffs0, s.x0))
        amp0 = max(s.amp, float(np.interp(s.x0, x, y) - baseline_at), 0.1 * yspan)
        gamma0 = max(0.10, min(abs(s.gamma), 0.35 * span))

        p0.extend([amp0, s.x0, gamma0])

        lo.extend([0.0, s.x0 - max(0.8, 0.15 * span), 0.03])
        hi.extend([20.0 * yspan, s.x0 + max(0.8, 0.15 * span), 0.50 * span])

    return np.array(p0, float), np.array(lo, float), np.array(hi, float), seeds


def fit_stage_a(
    x: np.ndarray,
    y: np.ndarray,
    core: CoreSeedBundle,
    baseline_deg: int = 2,
    max_peaks: Optional[int] = None,
) -> Dict[str, Any]:
    p0, lo, hi, used_seeds = build_initial_stage_a_guess(
        x=x, y=y, core=core, baseline_deg=baseline_deg, max_peaks=max_peaks
    )
    n_peaks = len(used_seeds)

    def residuals(p: np.ndarray) -> np.ndarray:
        yhat, _, _ = eval_stage_a_model(x, p, baseline_deg=baseline_deg, n_peaks=n_peaks)
        return (yhat - y)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lo, hi),
        method="trf",
        loss="soft_l1",
        f_scale=max(core.sigma_noise, robust_mad_sigma(y) * 0.5),
        max_nfev=20000,
    )

    p = sol.x
    yhat, baseline, comps = eval_stage_a_model(x, p, baseline_deg=baseline_deg, n_peaks=n_peaks)
    resid = y - yhat

    peaks = []
    idx = baseline_deg + 1
    for i in range(n_peaks):
        amp, x0, gamma = p[idx : idx + 3]
        peaks.append(
            {
                "i": i + 1,
                "amp": float(amp),
                "x0": float(x0),
                "gamma": float(abs(gamma)),
                "height_proxy": float(lorentz(np.array([x0]), amp, x0, abs(gamma))[0]),
            }
        )
        idx += 3

    # sort by center
    peaks = sorted(peaks, key=lambda d: d["x0"])

    k = len(p)
    out = {
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "params": p.tolist(),
        "baseline_coeffs": p[: baseline_deg + 1].tolist(),
        "peaks": peaks,
        "fit_y": yhat,
        "baseline_y": baseline,
        "components": comps,
        "residual": resid,
        "rmse": rmse(y, yhat),
        "bic": bic_score(y, yhat, k),
        "aic": aic_score(y, yhat, k),
        "resid_sigma": robust_mad_sigma(resid),
    }
    return out


# ---------------------------------------------------------------------
# Anchored RTIM proxy
# ---------------------------------------------------------------------

def build_rtim_proxy(
    x: np.ndarray,
    stage_a: Dict[str, Any],
    sigma_noise: float,
) -> Dict[str, np.ndarray]:
    peaks = stage_a["peaks"]
    comps = stage_a["components"]

    if not peaks:
        raise RuntimeError("No peaks available from Stage-A fit.")

    amps = np.array([max(p["amp"], 1e-12) for p in peaks], dtype=float)
    centers = np.array([p["x0"] for p in peaks], dtype=float)
    gammas = np.array([max(p["gamma"], 1e-6) for p in peaks], dtype=float)

    weights = amps / amps.sum()
    omega0_ref = float(np.sum(weights * centers))
    gamma_ref = float(np.sum(weights * gammas))

    L_core = np.sum(np.vstack(comps), axis=0)
    L_core = np.maximum(L_core, 0.0)

    L_norm = L_core / max(float(L_core.max()), 1e-12)

    # Simple, stable proxy parameters derived from real core
    c_s = float(max(np.max(L_norm), 1e-6))
    K_p = float(0.8)  # stable, mild saturation
    alpha_ss = c_s * L_norm
    Gamma_tr = alpha_ss / (1.0 + K_p * alpha_ss)

    # A benign transfer proxy – not a claim of realized fusion
    E_pair = Gamma_tr / (max(gamma_ref, 1e-9) + Gamma_tr)

    R_core = E_pair / max(sigma_noise, 1e-12)

    return {
        "omega0_ref": np.full_like(x, omega0_ref, dtype=float),
        "gamma_ref": np.full_like(x, gamma_ref, dtype=float),
        "L_core": L_norm,
        "alpha_ss": alpha_ss,
        "Gamma_tr": Gamma_tr,
        "E_pair": E_pair,
        "R_core": R_core,
        "sigma_noise": np.full_like(x, sigma_noise, dtype=float),
        "c_s": np.full_like(x, c_s, dtype=float),
        "K_p": np.full_like(x, K_p, dtype=float),
    }


# ---------------------------------------------------------------------
# Plotting / export
# ---------------------------------------------------------------------

def export_core_table(path: str, stage_a: Dict[str, Any]) -> None:
    df = pd.DataFrame(stage_a["peaks"])
    df.to_csv(path, index=False)


def make_plot(
    out_png: str,
    x: np.ndarray,
    y: np.ndarray,
    stage_a: Dict[str, Any],
    proxy: Dict[str, np.ndarray],
    title: str,
) -> None:
    fit_y = stage_a["fit_y"]
    baseline_y = stage_a["baseline_y"]
    comps = stage_a["components"]
    peaks = stage_a["peaks"]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.0, 0.8], hspace=0.28)

    # Panel 1: real data + stable fit
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, s=10, alpha=0.75, label="real data")
    ax1.plot(x, fit_y, lw=2.5, label="stable Stage-A fit")
    ax1.plot(x, baseline_y, "--", lw=2.0, label="baseline")

    colors = ["tab:red", "tab:purple", "tab:green", "tab:orange", "tab:brown"]
    for i, comp in enumerate(comps):
        color = colors[i % len(colors)]
        ax1.plot(x, baseline_y + comp, ":", lw=2.0, color=color, label=f"mode {i+1}")
        px = peaks[i]["x0"]
        ax1.axvline(px, color=color, alpha=0.18)

    ax1.set_title(title)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Signal")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.25)

    # Panel 2: anchored RTIM proxy
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(x, proxy["L_core"], lw=2.0, label="L_core")
    ax2.plot(x, proxy["Gamma_tr"], lw=2.0, label="Gamma_tr")
    ax2.plot(x, proxy["E_pair"], lw=2.0, label="E_pair proxy")
    ax2.set_title("Anchored RTIM proxy over the real core")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Proxy amplitude")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.25)

    # Panel 3: detectability
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(x, proxy["R_core"], lw=2.2, label="R_core = E_pair / sigma_noise")
    for p in peaks:
        ax3.axvline(p["x0"], color="gray", alpha=0.22)
    ax3.set_title("Readout / detectability proxy")
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Detectability")
    ax3.legend(loc="best")
    ax3.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


# --- upgrade ---
def _to_numeric_1d(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float).squeeze()
    except Exception:
        return None
    if arr.ndim == 1 and arr.size >= 8 and np.all(np.isfinite(arr)):
        return arr.astype(float).ravel()
    return None


def _is_numeric_1d(x: Any) -> bool:
    return _to_numeric_1d(x) is not None


def _try_curve_from_dict(d: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    keymap = {str(k).lower(): k for k in d.keys()}

    x_keys = [
        "x", "freq", "frequency", "omega", "xdata", "f",
        "ppm", "energy", "time", "t"
    ]
    y_keys = [
        "y", "signal", "counts", "amplitude", "amp", "z",
        "zss", "z_ss", "ydata", "intensity", "value", "values"
    ]

    # 1) preferred named pairing
    x = None
    y = None

    for k in x_keys:
        if k in keymap:
            arr = _to_numeric_1d(d[keymap[k]])
            if arr is not None:
                x = arr
                break

    for k in y_keys:
        if k in keymap:
            arr = _to_numeric_1d(d[keymap[k]])
            if arr is not None:
                y = arr
                break

    if x is not None and y is not None and len(x) == len(y):
        return x, y

    # 2) fallback: pair any two same-length numeric 1D fields
    numeric_fields: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        arr = _to_numeric_1d(v)
        if arr is not None:
            numeric_fields[str(k)] = arr

    items = list(numeric_fields.items())
    for i, (kx, xv) in enumerate(items):
        dx = np.diff(xv)
        monotonic = np.all(dx >= 0) or np.all(dx <= 0)
        unique_ratio = np.unique(xv).size / max(len(xv), 1)

        # x candidate should look axis-like
        if not monotonic or unique_ratio < 0.8:
            continue

        for ky, yv in items:
            if ky == kx:
                continue
            if len(xv) != len(yv):
                continue
            if np.std(yv) <= 0:
                continue
            return xv, yv

    return None


def _try_curve_from_array(arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(arr)

    # structured array fallback
    if arr.dtype.names:
        as_dict = {name: arr[name] for name in arr.dtype.names}
        return _try_curve_from_dict(as_dict)

    if not np.issubdtype(arr.dtype, np.number):
        return None

    arr = np.squeeze(arr)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            x = np.asarray(arr[:, 0], dtype=float).ravel()
            y = np.asarray(arr[:, 1], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y
        if arr.shape[0] >= 2:
            x = np.asarray(arr[0, :], dtype=float).ravel()
            y = np.asarray(arr[1, :], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y

    return None

def _to_numeric_1d(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float).squeeze()
    except Exception:
        return None
    if arr.ndim == 1 and arr.size >= 8 and np.all(np.isfinite(arr)):
        return arr.astype(float).ravel()
    return None


def _is_numeric_1d(x: Any) -> bool:
    return _to_numeric_1d(x) is not None


def _try_curve_from_dict(d: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    keymap = {str(k).lower(): k for k in d.keys()}

    x_keys = [
        "x", "freq", "frequency", "omega", "xdata", "f",
        "ppm", "energy", "time", "t"
    ]
    y_keys = [
        "y", "signal", "counts", "amplitude", "amp", "z",
        "zss", "z_ss", "ydata", "intensity", "value", "values"
    ]

    # 1) preferred named pairing
    x = None
    y = None

    for k in x_keys:
        if k in keymap:
            arr = _to_numeric_1d(d[keymap[k]])
            if arr is not None:
                x = arr
                break

    for k in y_keys:
        if k in keymap:
            arr = _to_numeric_1d(d[keymap[k]])
            if arr is not None:
                y = arr
                break

    if x is not None and y is not None and len(x) == len(y):
        return x, y

    # 2) fallback: pair any two same-length numeric 1D fields
    numeric_fields: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        arr = _to_numeric_1d(v)
        if arr is not None:
            numeric_fields[str(k)] = arr

    items = list(numeric_fields.items())
    for i, (kx, xv) in enumerate(items):
        dx = np.diff(xv)
        monotonic = np.all(dx >= 0) or np.all(dx <= 0)
        unique_ratio = np.unique(xv).size / max(len(xv), 1)

        # x candidate should look axis-like
        if not monotonic or unique_ratio < 0.8:
            continue

        for ky, yv in items:
            if ky == kx:
                continue
            if len(xv) != len(yv):
                continue
            if np.std(yv) <= 0:
                continue
            return xv, yv

    return None


def _try_curve_from_array(arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(arr)

    # structured array fallback
    if arr.dtype.names:
        as_dict = {name: arr[name] for name in arr.dtype.names}
        return _try_curve_from_dict(as_dict)

    if not np.issubdtype(arr.dtype, np.number):
        return None

    arr = np.squeeze(arr)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            x = np.asarray(arr[:, 0], dtype=float).ravel()
            y = np.asarray(arr[:, 1], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y
        if arr.shape[0] >= 2:
            x = np.asarray(arr[0, :], dtype=float).ravel()
            y = np.asarray(arr[1, :], dtype=float).ravel()
            if len(x) >= 8 and len(y) == len(x):
                return x, y

    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Anchored FUZE / RTIM over real spectral data")
    ap.add_argument("--core-summary", required=True, help="Path to core_summary.json from the safe extractor")
    ap.add_argument("--spectrum", required=True, help="Path to .mat or .csv spectrum file")
    ap.add_argument("--dataset-key", default=None, help="Substring for MAT dataset selection")
    ap.add_argument("--x-col", type=int, default=0, help="CSV x column index")
    ap.add_argument("--y-col", type=int, default=1, help="CSV y column index")
    ap.add_argument("--window-min", type=float, default=None, help="Optional override for fit window min")
    ap.add_argument("--window-max", type=float, default=None, help="Optional override for fit window max")
    ap.add_argument("--baseline-deg", type=int, default=2, help="Polynomial baseline degree")
    ap.add_argument("--max-peaks", type=int, default=None, help="Use only first N seeded core peaks")
    ap.add_argument("--out-prefix", default="fuze_abc_real_v1", help="Output prefix")

    args = ap.parse_args()

    print("=== FUZE / RTIM Anchored ABC Real v1 ===")
    print(f"Loading core summary: {args.core_summary}")
    core = parse_core_summary(args.core_summary)

    print(f"Loading spectrum: {args.spectrum}")
    x, y, dataset_name = load_spectrum(args.spectrum, args.dataset_key, args.x_col, args.y_col)
    print(f"Selected dataset: {dataset_name}")
    print(f"Loaded points: {len(x)}")

    wmin = args.window_min if args.window_min is not None else core.window_min
    wmax = args.window_max if args.window_max is not None else core.window_max

    if wmin is not None and wmax is not None:
        mask = (x >= wmin) & (x <= wmax)
        if mask.sum() < 20:
            raise RuntimeError("Fit window is too small after masking.")
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x.copy()
        y_fit = y.copy()

    print(f"Window: [{x_fit.min():.4f}, {x_fit.max():.4f}]")
    print(f"transfer_supported in seed summary: {core.transfer_supported}")

    stage_a = fit_stage_a(
        x=x_fit,
        y=y_fit,
        core=core,
        baseline_deg=args.baseline_deg,
        max_peaks=args.max_peaks,
    )

    proxy = build_rtim_proxy(
        x=x_fit,
        stage_a=stage_a,
        sigma_noise=max(core.sigma_noise, stage_a["resid_sigma"]),
    )

    out_summary = f"{args.out_prefix}_summary.json"
    out_table = f"{args.out_prefix}_core_table.csv"
    out_plot = f"{args.out_prefix}_plot.png"

    export_core_table(out_table, stage_a)

    summary = {
        "data_source": args.spectrum,
        "dataset_name": dataset_name,
        "window": [float(x_fit.min()), float(x_fit.max())],
        "transfer_supported_seed": bool(core.transfer_supported),
        "stage_a_success": bool(stage_a["success"]),
        "stage_a_message": str(stage_a["message"]),
        "stage_a_rmse": float(stage_a["rmse"]),
        "stage_a_bic": float(stage_a["bic"]),
        "stage_a_aic": float(stage_a["aic"]),
        "stage_a_resid_sigma": float(stage_a["resid_sigma"]),
        "sigma_noise_used": float(max(core.sigma_noise, stage_a["resid_sigma"])),
        "core_fit": {
            "baseline_coeffs": [float(v) for v in stage_a["baseline_coeffs"]],
            "peaks": stage_a["peaks"],
        },
        "rtim_proxy": {
            "omega0_ref": float(proxy["omega0_ref"][0]),
            "gamma_ref": float(proxy["gamma_ref"][0]),
            "c_s": float(proxy["c_s"][0]),
            "K_p": float(proxy["K_p"][0]),
            "R_core_peak": float(np.max(proxy["R_core"])),
            "R_core_peak_x": float(x_fit[np.argmax(proxy["R_core"])]),
            "E_pair_peak": float(np.max(proxy["E_pair"])),
            "E_pair_peak_x": float(x_fit[np.argmax(proxy["E_pair"])]),
        },
        "verdict": {
            "note": (
                "This is an anchored spectral + RTIM-proxy fit. "
                "It is a disciplined model step, not an experimental proof of realized fusion."
            ),
            "transfer_supported": bool(core.transfer_supported),
            "primary_path_ok": bool(stage_a["success"]),
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    title = "FUZE / RTIM — anchored real-data Stage-A + RTIM proxy"
    make_plot(out_plot, x_fit, y_fit, stage_a, proxy, title)

    print("\n=== RESULT ===")
    print(f"Stage-A success      : {stage_a['success']}")
    print(f"Stage-A RMSE         : {stage_a['rmse']:.6g}")
    print(f"Stage-A BIC          : {stage_a['bic']:.6f}")
    print(f"residual sigma       : {stage_a['resid_sigma']:.6g}")
    print(f"omega0_ref           : {proxy['omega0_ref'][0]:.6f}")
    print(f"gamma_ref            : {proxy['gamma_ref'][0]:.6f}")
    print(f"R_core peak          : {np.max(proxy['R_core']):.6f}")
    print(f"R_core peak x        : {x_fit[np.argmax(proxy['R_core'])]:.6f}")
    print(f"transfer_supported   : {core.transfer_supported}")

    print("\nSaved:")
    print(f"  {out_summary}")
    print(f"  {out_table}")
    print(f"  {out_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())