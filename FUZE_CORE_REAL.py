#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
FUZE_CORE_REAL.py
-----------------
Core RTIM/Stage-A runner over real spectral data.

Purpose:
- takes the public Dataverse preflight report (fit_report.json),
- extracts the physically interesting *core* resonances,
- optionally refits them on real x/y data from clean_spectrum.csv,
- builds an RTIM-style transfer proxy anchored to those real peaks,
- emits a compact scorecard + plot.

This is intentionally brutal and narrow:
- it does NOT claim a full A/B/C proof,
- it gives you a hard real-data resonance core to build on.

Typical workflow:
1) Use FUZE_BUH_v2.py to make fit_report.json + clean_spectrum.csv
2) Run:
   python FUZE_CORE_REAL.py --report fit_report.json --csv clean_spectrum.csv --outdir core_out

Fallback report-only mode:
   python FUZE_CORE_REAL.py --report fit_report.json --outdir core_out

If only the report is available, the script still extracts the core peaks
and builds the RTIM proxy on a synthetic x-grid spanning the fit window.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares


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

    # Score = prefer small gamma and not too close to edges.
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
        data = loadmat(path)
        arrays = []
        for k, v in data.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                arrays.append((k, np.squeeze(v)))
        # best case: one 2-column array
        for name, arr in arrays:
            if arr.ndim == 2 and min(arr.shape) == 2:
                arr2 = arr if arr.shape[1] == 2 else arr.T
                x = np.asarray(arr2[:, 0], dtype=float)
                y = np.asarray(arr2[:, 1], dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                order = np.argsort(x[m])
                return x[m][order], y[m][order], f"{name}[0]", f"{name}[1]"
        # fallback: two longest 1D numeric arrays
        one_d = [(k, np.ravel(v)) for k, v in arrays if np.ravel(v).ndim == 1 and np.ravel(v).size >= 10]
        one_d.sort(key=lambda kv: kv[1].size, reverse=True)
        if len(one_d) >= 2:
            x = np.asarray(one_d[0][1], dtype=float)
            y = np.asarray(one_d[1][1], dtype=float)
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            order = np.argsort(x[m])
            return x[m][order], y[m][order], one_d[0][0], one_d[1][0]
        raise RuntimeError("MAT soubor jsem neuměl rozumně rozbalit na x/y.")
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
        A, x0, gamma = theta[idx : idx + 3]
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
    # crude baseline seed from low-percentile points
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
# RTIM proxy over the real core
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
# Statistics / diagnostics
# ----------------------------


def information_criteria(y: np.ndarray, yhat: np.ndarray, k: int) -> Tuple[float, float, float]:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rmse = float(np.sqrt(rss / max(n, 1)))
    rss_n = max(rss / max(n, 1), 1e-300)
    aic = float(n * np.log(rss_n) + 2 * k)
    bic = float(n * np.log(rss_n) + k * np.log(max(n, 1)))
    return rmse, aic, bic


# ----------------------------
# Plotting
# ----------------------------


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
        ax1.plot(x, yhat, "-", lw=2.2, label="core fit")
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

def safe_plot_label(label: str) -> str:
    label = str(label)

    # u těchto Dataverse názvů bývá za "/" už to důležité
    # např. "Sample B, ... /Frequency (GHz)" -> "Frequency (GHz)"
    if "/" in label:
        label = label.split("/")[-1]

    # vypni mathtext tím, že odstraníš problematické LaTeX znaky
    label = label.replace("$", "")
    label = label.replace(r"\times", "×")
    label = label.replace(r"\mathcal{", "")
    label = label.replace("{", "")
    label = label.replace("}", "")

    return label.strip()

# ----------------------------
# Main runner
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Brutální core RTIM Stage-A runner over real spectral data")
    ap.add_argument("--report", default="fit_report.json", help="path to fit_report.json")
    ap.add_argument("--csv", default="fig2c_data.mat", help="path to clean_spectrum.csv / tsv / xlsx / mat")
    ap.add_argument("--xcol", default=None)
    ap.add_argument("--ycol", default=None)
    ap.add_argument("--degree", type=int, default=1, choices=[0, 1, 2], help="baseline degree for core refit")
    ap.add_argument("--outdir", default="core_out")
    ap.add_argument("--max-nfev", type=int, default=4000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = load_report(Path(args.report))
    xmin, xmax = _window_from_report(report)
    core_peaks_seed = choose_core_peaks(report, n_core=2)
    core_window = core_window_from_peaks(core_peaks_seed, xmin=xmin, xmax=xmax, pad_factor=2.0)

    summary: Dict = {
        "report_path": str(Path(args.report).resolve()),
        "fit_window": {"xmin": xmin, "xmax": xmax},
        "best_model_BIC": report["best_model_by_BIC"]["BIC"],
        "best_model_n_peaks": report["best_model_by_BIC"]["n_peaks"],
        "core_seed_peaks": [asdict(p) | {"Q": p.Q} for p in core_peaks_seed],
        "core_window": {"xmin": core_window[0], "xmax": core_window[1]},
    }

    if args.csv:
        x, y, xc, yc = load_xy(Path(args.csv), xcol=args.xcol, ycol=args.ycol)
        xfit, yfit, theta = fit_core_window(x, y, core_peaks_seed, degree=args.degree, x_window=core_window, max_nfev=args.max_nfev)
        x_ref = float(np.mean(xfit))
        yhat, baseline, comps = core_model(xfit, theta, degree=args.degree, n_peaks=len(core_peaks_seed), x_ref=x_ref)
        coeffs, peaks_fit = pack_core_params(theta, degree=args.degree, n_peaks=len(core_peaks_seed))

        sigma_noise = mad_sigma(yfit - yhat)
        proxy = build_rtim_proxy(xfit, peaks_fit, sigma_noise=sigma_noise)
        proxy_peak_x = float(xfit[int(np.argmax(proxy["Rcore"]))])
        narrow_centers = [p.x0 for p in peaks_fit]
        peak_alignment_error = float(min(abs(proxy_peak_x - c) for c in narrow_centers))
        peak_alignment_ok = bool(peak_alignment_error <= max(0.35, 0.20 * min(p.fwhm for p in peaks_fit)))
        rmse, aic, bic = information_criteria(yfit, yhat, k=len(theta))

        result = CoreFitResult(
            success=True,
            message="core refit completed",
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
            }
        )

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
            title="FUZE / RTIM core fit over real data",
        )
    else:
        # report-only mode: build a synthetic x-grid and use the seeds directly.
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
