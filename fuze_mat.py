#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUZE canonical runner
=====================

What this script does
---------------------
1. Loads a spectrum either from a local file or from Harvard Dataverse via DOI.
2. Tries to identify sensible x/y columns in CSV/TSV/TXT/MAT data.
3. Cleans numeric data, optionally crops the x-range, and saves clean_spectrum.csv.
4. Runs a conservative multipeak Lorentzian + polynomial baseline model search.
5. Scores candidate models with AIC/BIC and exports fit_report.json + fit_plot.png.
6. Builds a compact "core" readout and exports:
       core_summary.json
       core_summary.txt
       core_table.csv
       core_real_plot.png

Design goal
-----------
This is meant to be the canonical entrypoint for the current FUZE repository.
It merges the public-data ingest from the BUH scripts with a safe post-fit
summary layer inspired by the CORE_REAL_v2_safe branch.

Example
-------
    python fuze.py --file spectrum.csv
    python fuze.py --doi doi:10.7910/DVN/XCS15A --output-dir out
    python fuze.py --file data.mat --xcol frequency --ycol counts --max-peaks 3



    D:/HIT/PythonProject/.venv/Scripts/python fuze_mat.py --file fig2c_data.mat   --mat-key 'Sample A, 5$\mathcal{\times10^{13} cm^{-2}}$'   --output-dir D:\hit\PythonProject\   --max-peaks 3   --max-baseline-degree 2

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateFit:
    n_peaks: int
    baseline_degree: int
    success: bool
    params: List[float]
    param_names: List[str]
    sse: float
    rmse: float
    aic: float
    bic: float
    r2: float
    peak_count_detected: int
    message: str


# ---------------------------------------------------------------------------
# Dataverse ingest
# ---------------------------------------------------------------------------

def fetch_dataset_metadata(doi: str, timeout: int = 60) -> Dict[str, Any]:
    params = {"persistentId": doi}
    response = requests.get(DATAVERSE_META_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API error: {payload}")
    return payload["data"]


def extract_files(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest = meta.get("latestVersion", {})
    files = latest.get("files", [])
    out: List[Dict[str, Any]] = []
    for item in files:
        data_file = item.get("dataFile", {})
        out.append(
            {
                "id": data_file.get("id"),
                "pid": data_file.get("persistentId"),
                "filename": data_file.get("filename"),
                "contentType": data_file.get("contentType"),
                "filesize": data_file.get("filesize"),
                "description": item.get("description", ""),
                "tabular": bool(data_file.get("tabularData")),
            }
        )
    return out


def choose_candidate_file(
    files: List[Dict[str, Any]],
    file_name_hint: Optional[str] = None,
) -> Dict[str, Any]:
    if not files:
        raise RuntimeError("Dataset neobsahuje žádné soubory.")

    if file_name_hint:
        hint = file_name_hint.lower()
        hinted = [f for f in files if hint in str(f.get("filename", "")).lower()]
        if hinted:
            return hinted[0]

    def score(item: Dict[str, Any]) -> Tuple[int, int]:
        name = str(item.get("filename") or "").lower()
        ctype = str(item.get("contentType") or "").lower()
        score_value = 0
        if item.get("tabular"):
            score_value += 100
        if name.endswith(".csv"):
            score_value += 50
        if name.endswith(".tsv"):
            score_value += 45
        if name.endswith(".txt"):
            score_value += 30
        if name.endswith(".dat"):
            score_value += 20
        if name.endswith(".mat"):
            score_value += 25
        if "csv" in ctype:
            score_value += 40
        if "tab-separated" in ctype:
            score_value += 35
        if "text/plain" in ctype:
            score_value += 15
        filesize = int(item.get("filesize") or 10**12)
        return score_value, -filesize

    ranked = sorted(files, key=score, reverse=True)
    best = ranked[0]
    if score(best)[0] <= 0:
        raise RuntimeError("Nenašel jsem rozumný datový soubor. Použij --file-name.")
    return best


def download_file_bytes(file_id: int, timeout: int = 120) -> bytes:
    url = DATAVERSE_FILE_URL.format(file_id=file_id)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Table / MAT parsing
# ---------------------------------------------------------------------------

def _collect_h5_datasets(h5obj: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in h5obj.keys():
        item = h5obj[key]
        name = f"{prefix}/{key}" if prefix else key
        if h5py is not None and isinstance(item, h5py.Dataset):
            try:
                out[name] = np.array(item)
            except Exception:
                pass
        elif h5py is not None and isinstance(item, h5py.Group):
            out.update(_collect_h5_datasets(item, prefix=name))
    return out


def _flatten_numeric_array(arr: Any) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    return arr


def _score_key(name: str, patterns: Sequence[str]) -> int:
    lower = name.lower()
    return sum(10 for pat in patterns if pat in lower)


def _extract_numeric_candidates(value: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    if hasattr(value, "_fieldnames"):
        for field in getattr(value, "_fieldnames", []) or []:
            child = getattr(value, field)
            child_prefix = f"{prefix}/{field}" if prefix else str(field)
            out.update(_extract_numeric_candidates(child, child_prefix))
        return out

    arr = np.asarray(value)

    if getattr(arr.dtype, "names", None):
        for field in arr.dtype.names or ():
            child_prefix = f"{prefix}/{field}" if prefix else str(field)
            out.update(_extract_numeric_candidates(arr[field], child_prefix))
        return out

    if arr.dtype == object:
        flat = np.ravel(arr)
        if flat.size == 1:
            return _extract_numeric_candidates(flat[0], prefix)
        for idx, item in enumerate(flat):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"item_{idx}"
            out.update(_extract_numeric_candidates(item, child_prefix))
        return out

    numeric = _flatten_numeric_array(arr)
    if numeric is not None and prefix:
        out[prefix] = numeric
    return out


def _mat_dict_to_dataframe(
    mat_dict: Dict[str, Any], filename: str = "", mat_key: Optional[str] = None
) -> pd.DataFrame:
    candidates: Dict[str, np.ndarray] = {}
    for key, value in mat_dict.items():
        if str(key).startswith("__"):
            continue
        if mat_key is not None and str(key) != mat_key:
            continue
        extracted = _extract_numeric_candidates(value, prefix=str(key))
        candidates.update(extracted)

    if not candidates:
        raise RuntimeError(f"Soubor '{filename}' neobsahuje čitelné numerické MATLAB proměnné.")

    matrix_candidates: List[Tuple[str, np.ndarray]] = []
    for key, arr in candidates.items():
        if arr.ndim == 2 and min(arr.shape) == 2 and max(arr.shape) >= 10:
            matrix_candidates.append((key, arr))
    if matrix_candidates:
        key, arr = sorted(matrix_candidates, key=lambda kv: kv[1].size, reverse=True)[0]
        if arr.shape[1] == 2:
            return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
        if arr.shape[0] == 2:
            return pd.DataFrame({"x": arr[0, :], "y": arr[1, :]})

    vector_candidates = {
        key: arr for key, arr in candidates.items() if arr.ndim == 1 and len(arr) >= 10
    }
    if len(vector_candidates) >= 2:
        x_patterns = ["x", "freq", "frequency", "omega", "detuning", "time", "energy"]
        y_patterns = ["y", "count", "counts", "signal", "intensity", "power", "trans"]
        keys = list(vector_candidates)
        best_pair: Optional[Tuple[str, str]] = None
        best_score: Optional[Tuple[int, int, int]] = None
        for kx in keys:
            for ky in keys:
                if kx == ky:
                    continue
                x_arr = vector_candidates[kx]
                y_arr = vector_candidates[ky]
                if len(x_arr) != len(y_arr):
                    continue
                score = (_score_key(kx, x_patterns), _score_key(ky, y_patterns), len(x_arr))
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (kx, ky)
        if best_pair is not None:
            kx, ky = best_pair
            return pd.DataFrame({kx: vector_candidates[kx], ky: vector_candidates[ky]})

    vector_items = list(vector_candidates.items())
    for i in range(len(vector_items)):
        for j in range(i + 1, len(vector_items)):
            kx, x_arr = vector_items[i]
            ky, y_arr = vector_items[j]
            if len(x_arr) == len(y_arr):
                return pd.DataFrame({kx: x_arr, ky: y_arr})

    shapes = {key: tuple(np.asarray(value).shape) for key, value in candidates.items()}
    raise RuntimeError(
        f"Nepodařilo se automaticky rozpoznat x/y data v '{filename}'. Dostupné tvary: {shapes}"
    )


def load_mat_from_bytes(
    data: bytes, filename: str, mat_key: Optional[str] = None
) -> pd.DataFrame:
    try:
        bio = io.BytesIO(data)
        mat = loadmat(bio, squeeze_me=True, struct_as_record=False)
        return _mat_dict_to_dataframe(mat, filename=filename, mat_key=mat_key)
    except NotImplementedError:
        pass
    except Exception:
        pass

    if h5py is None:
        raise RuntimeError(
            f"Soubor '{filename}' vypadá jako MATLAB v7.3/HDF5, ale chybí balík h5py."
        )

    try:
        bio = io.BytesIO(data)
        with h5py.File(bio, "r") as handle:
            h5dict = _collect_h5_datasets(handle)
        return _mat_dict_to_dataframe(h5dict, filename=filename, mat_key=mat_key)
    except Exception as exc:
        raise RuntimeError(
            f"Nepodařilo se načíst MATLAB soubor '{filename}' ani přes scipy.io.loadmat, ani přes h5py. Detail: {exc}"
        ) from exc


def sniff_delimiter(sample: str, fallback: str = ",") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return str(dialect.delimiter)
    except Exception:
        return fallback


def load_table_from_bytes(
    data: bytes, filename: str, mat_key: Optional[str] = None
) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext == ".mat":
        return load_mat_from_bytes(data, filename=filename, mat_key=mat_key)

    text = data.decode("utf-8", errors="replace")
    sample = text[:5000]
    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = sniff_delimiter(sample, fallback=",")
    else:
        sep = sniff_delimiter(sample, fallback=",")

    parsers = [
        lambda: pd.read_csv(io.StringIO(text), sep=sep),
        lambda: pd.read_csv(io.StringIO(text), sep=None, engine="python"),
        lambda: pd.read_csv(io.StringIO(text), sep=sep, header=None),
    ]
    for parser in parsers:
        try:
            df = parser()
            if df.shape[1] >= 2:
                if all(str(c).isdigit() for c in df.columns):
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                return df
        except Exception:
            continue

    raise RuntimeError(f"Nepodařilo se načíst tabulku ze souboru '{filename}'.")


def load_local_table(path: Path, mat_key: Optional[str] = None) -> pd.DataFrame:
    data = path.read_bytes()
    return load_table_from_bytes(data, filename=path.name, mat_key=mat_key)


def clean_numeric_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("\u2212", "-", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\dEe\+\-\.]", "", regex=True)
            )
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def guess_xy_columns(
    df: pd.DataFrame,
    xcol: Optional[str] = None,
    ycol: Optional[str] = None,
) -> Tuple[str, str]:
    if xcol and ycol:
        if xcol not in df.columns or ycol not in df.columns:
            raise RuntimeError(f"Zadané sloupce '{xcol}' / '{ycol}' v datech nejsou.")
        return xcol, ycol

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise RuntimeError("Nenašel jsem aspoň dva numerické sloupce.")

    name_scores: List[Tuple[int, str]] = []
    x_patterns = ["x", "freq", "frequency", "omega", "detuning", "energy", "time"]
    y_patterns = ["y", "count", "counts", "signal", "intensity", "trans", "power"]
    for col in numeric_cols:
        lower = str(col).lower()
        score = 0
        if any(p in lower for p in x_patterns):
            score += 20
        if any(p in lower for p in y_patterns):
            score -= 20
        name_scores.append((score, str(col)))

    x_guess = max(name_scores)[1]
    remaining = [c for c in numeric_cols if c != x_guess]
    y_guess = remaining[0]

    # If the chosen x is not monotone enough, fall back to first monotone numeric column.
    monotone_candidates = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(values) >= 10:
            frac_monotone = max(
                np.mean(np.diff(values) >= 0),
                np.mean(np.diff(values) <= 0),
            )
            monotone_candidates.append((float(frac_monotone), str(col)))
    if monotone_candidates and max(monotone_candidates)[0] >= 0.9:
        mono_col = max(monotone_candidates)[1]
        if mono_col != y_guess:
            x_guess = mono_col
            remaining = [c for c in numeric_cols if c != x_guess]
            y_guess = remaining[0]

    return x_guess, y_guess


def prepare_xy_dataframe(
    df: pd.DataFrame,
    xcol: Optional[str],
    ycol: Optional[str],
    xmin: Optional[float],
    xmax: Optional[float],
) -> pd.DataFrame:
    clean = clean_numeric_series(df)
    x_name, y_name = guess_xy_columns(clean, xcol=xcol, ycol=ycol)
    out = clean[[x_name, y_name]].rename(columns={x_name: "x", y_name: "y"})
    out = out.dropna().copy()
    out = out[np.isfinite(out["x"]) & np.isfinite(out["y"])]
    out = out.sort_values("x").drop_duplicates(subset=["x"], keep="first")
    if xmin is not None:
        out = out[out["x"] >= xmin]
    if xmax is not None:
        out = out[out["x"] <= xmax]
    if len(out) < 20:
        raise RuntimeError("Po vyčištění zůstalo příliš málo bodů pro smysluplný fit.")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model family: polynomial baseline + n Lorentz peaks
# ---------------------------------------------------------------------------

def polynomial_baseline(x: np.ndarray, coeffs: Sequence[float]) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for power, coeff in enumerate(coeffs):
        y += coeff * x ** power
    return y


def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-12)
    return amplitude * (gamma**2 / ((x - center) ** 2 + gamma**2))


def model_value(x: np.ndarray, params: Sequence[float], n_peaks: int, baseline_degree: int) -> np.ndarray:
    n_base = baseline_degree + 1
    coeffs = params[:n_base]
    y = polynomial_baseline(x, coeffs)
    offset = n_base
    for _ in range(n_peaks):
        amp, ctr, gam = params[offset:offset + 3]
        y += lorentzian(x, amp, ctr, gam)
        offset += 3
    return y


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    deg = min(max(degree, 0), max(0, len(x) - 1))
    coeff_desc = np.polyfit(x, y, deg=deg)
    coeff_asc = coeff_desc[::-1]
    if len(coeff_asc) < degree + 1:
        coeff_asc = np.pad(coeff_asc, (0, degree + 1 - len(coeff_asc)))
    return coeff_asc.astype(float)


def initial_peak_seeds(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int,
    prominence_fraction: float,
) -> List[Tuple[float, float, float]]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    span_x = float(np.max(x) - np.min(x))
    y_span = float(np.max(y) - np.min(y))
    if span_x <= 0 or y_span <= 0:
        return []

    prominence = max(prominence_fraction * y_span, 1e-9)
    distance = max(1, len(x) // max(4 * max_peaks, 4))
    peak_idx, props = find_peaks(y, prominence=prominence, distance=distance)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(y))])
        prominences = np.array([y_span])
    else:
        prominences = np.asarray(props.get("prominences", np.ones_like(peak_idx)), dtype=float)

    order = np.argsort(prominences)[::-1]
    seeds: List[Tuple[float, float, float]] = []
    default_width = max(span_x / 40.0, 1e-6)
    for idx in peak_idx[order][:max_peaks]:
        amp = max(float(y[idx] - np.median(y)), y_span * 0.05)
        ctr = float(x[idx])
        gam = default_width
        seeds.append((amp, ctr, gam))
    return seeds


def build_initial_params(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    baseline_degree: int,
    prominence_fraction: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], List[str], int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_span = max(y_max - y_min, 1e-9)
    x_span = max(x_max - x_min, 1e-9)

    baseline = robust_polyfit(x, y, baseline_degree)
    base_names = [f"b{idx}" for idx in range(baseline_degree + 1)]

    seeds = initial_peak_seeds(x, y, max_peaks=n_peaks, prominence_fraction=prominence_fraction)
    if not seeds:
        seeds = [(y_span * 0.25, float(x[np.argmax(y)]), x_span / 40.0)]
    while len(seeds) < n_peaks:
        jitter = 0.1 * x_span * (len(seeds) + 1) / (n_peaks + 1)
        seeds.append((y_span * 0.15, float(np.median(x) + jitter), x_span / 40.0))
    seeds = seeds[:n_peaks]

    params: List[float] = list(map(float, baseline))
    lower: List[float] = [-np.inf] * len(base_names)
    upper: List[float] = [np.inf] * len(base_names)
    names = list(base_names)

    for i, (amp, ctr, gam) in enumerate(seeds, start=1):
        params.extend([amp, ctr, gam])
        lower.extend([-2.0 * y_span, x_min - 0.1 * x_span, x_span / 1e5])
        upper.extend([4.0 * max(y_span, abs(y_max), 1.0), x_max + 0.1 * x_span, 0.5 * x_span])
        names.extend([f"amp_{i}", f"center_{i}", f"gamma_{i}"])

    return np.asarray(params, dtype=float), (np.asarray(lower), np.asarray(upper)), names, len(seeds)


def safe_curve_fit(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    baseline_degree: int,
    prominence_fraction: float,
) -> CandidateFit:
    p0, bounds, names, detected = build_initial_params(
        x, y, n_peaks=n_peaks, baseline_degree=baseline_degree, prominence_fraction=prominence_fraction
    )

    def _wrapped_model(xval: np.ndarray, *params: float) -> np.ndarray:
        return model_value(xval, params=params, n_peaks=n_peaks, baseline_degree=baseline_degree)

    try:
        popt, _pcov = curve_fit(
            _wrapped_model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
        fit_y = _wrapped_model(x, *popt)
        success = True
        message = "ok"
        params = popt
    except Exception as exc:
        fit_y = _wrapped_model(x, *p0)
        success = False
        message = str(exc)
        params = p0

    residual = y - fit_y
    n = len(x)
    k = len(params)
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(sse / max(n, 1)))
    var = sse / max(n, 1)
    if not np.isfinite(var) or var <= 0:
        var = 1e-12
    aic = float(n * np.log(var) + 2 * k)
    bic = float(n * np.log(var) + k * np.log(max(n, 1)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan")

    return CandidateFit(
        n_peaks=n_peaks,
        baseline_degree=baseline_degree,
        success=success,
        params=[float(p) for p in params],
        param_names=names,
        sse=sse,
        rmse=rmse,
        aic=aic,
        bic=bic,
        r2=r2,
        peak_count_detected=int(detected),
        message=message,
    )


def rank_models(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int,
    max_baseline_degree: int,
    prominence_fraction: float,
) -> List[CandidateFit]:
    fits: List[CandidateFit] = []
    for degree in range(max_baseline_degree + 1):
        for n_peaks in range(1, max_peaks + 1):
            fit = safe_curve_fit(
                x,
                y,
                n_peaks=n_peaks,
                baseline_degree=degree,
                prominence_fraction=prominence_fraction,
            )
            fits.append(fit)
    fits.sort(key=lambda item: (math.isfinite(item.bic), -item.r2 if np.isfinite(item.r2) else np.inf), reverse=False)
    fits.sort(key=lambda item: item.bic if np.isfinite(item.bic) else np.inf)
    return fits


# ---------------------------------------------------------------------------
# Post-fit core layer
# ---------------------------------------------------------------------------

def unpack_fit_params(params: Sequence[float], n_peaks: int, baseline_degree: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    n_base = baseline_degree + 1
    coeffs = np.asarray(params[:n_base], dtype=float)
    peaks: List[Dict[str, float]] = []
    offset = n_base
    for _ in range(n_peaks):
        amp, ctr, gam = params[offset:offset + 3]
        peaks.append({"amplitude": float(amp), "center": float(ctr), "gamma": abs(float(gam))})
        offset += 3
    return coeffs, peaks


def evaluate_components(
    x: np.ndarray,
    params: Sequence[float],
    n_peaks: int,
    baseline_degree: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    coeffs, peaks = unpack_fit_params(params, n_peaks=n_peaks, baseline_degree=baseline_degree)
    baseline = polynomial_baseline(x, coeffs)
    components = [lorentzian(x, pk["amplitude"], pk["center"], pk["gamma"]) for pk in peaks]
    lcore = np.sum(components, axis=0) if components else np.zeros_like(x)
    return baseline, lcore, components


def peak_table(
    best_fit: CandidateFit,
    x: np.ndarray,
) -> pd.DataFrame:
    _baseline, _lcore, components = evaluate_components(
        x,
        params=best_fit.params,
        n_peaks=best_fit.n_peaks,
        baseline_degree=best_fit.baseline_degree,
    )
    coeffs, peaks = unpack_fit_params(best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    rows = []
    for idx, peak in enumerate(peaks, start=1):
        area_proxy = float(np.pi * peak["amplitude"] * peak["gamma"])
        height_proxy = float(np.max(components[idx - 1])) if components else float("nan")
        rows.append(
            {
                "peak_index": idx,
                "amplitude": peak["amplitude"],
                "center": peak["center"],
                "gamma": peak["gamma"],
                "fwhm_proxy": 2.0 * peak["gamma"],
                "area_proxy": area_proxy,
                "height_proxy": height_proxy,
                "baseline_degree": best_fit.baseline_degree,
            }
        )
    return pd.DataFrame(rows)


def compute_core_summary(
    df_xy: pd.DataFrame,
    best_fit: CandidateFit,
) -> Dict[str, Any]:
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    fit_y = model_value(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    baseline, lcore, components = evaluate_components(
        x,
        params=best_fit.params,
        n_peaks=best_fit.n_peaks,
        baseline_degree=best_fit.baseline_degree,
    )
    residual = y - fit_y

    residual_corr_with_lcore = float("nan")
    if np.std(residual) > 0 and np.std(lcore) > 0:
        residual_corr_with_lcore = float(np.corrcoef(residual, lcore)[0, 1])

    x_span = float(np.max(x) - np.min(x))
    peak_df = peak_table(best_fit, x)
    centers_inside = bool(peak_df["center"].between(np.min(x), np.max(x)).all()) if not peak_df.empty else False
    widths_reasonable = bool((peak_df["gamma"] > x_span / 1e5).all() and (peak_df["gamma"] < 0.5 * x_span).all()) if not peak_df.empty else False
    ordered_centers = bool(np.all(np.diff(peak_df["center"].to_numpy()) >= 0)) if len(peak_df) >= 2 else True
    peak_alignment_ok = bool(centers_inside and widths_reasonable and ordered_centers)

    transfer_supported = bool(best_fit.success and np.isfinite(best_fit.r2) and best_fit.r2 >= 0.80)
    residual_small = bool(best_fit.rmse <= 0.20 * max(np.std(y), 1e-12))

    strongest_idx = int(np.argmax(peak_df["area_proxy"].to_numpy())) if not peak_df.empty else -1
    strongest_peak = (
        peak_df.iloc[strongest_idx].to_dict() if strongest_idx >= 0 else None
    )

    summary: Dict[str, Any] = {
        "canonical_runner": "fuze.py",
        "n_points": int(len(df_xy)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "n_peaks": int(best_fit.n_peaks),
        "baseline_degree": int(best_fit.baseline_degree),
        "fit_success": bool(best_fit.success),
        "fit_message": best_fit.message,
        "rmse": float(best_fit.rmse),
        "r2": float(best_fit.r2),
        "aic": float(best_fit.aic),
        "bic": float(best_fit.bic),
        "transfer_supported": transfer_supported,
        "peak_alignment_ok": peak_alignment_ok,
        "residual_small": residual_small,
        "residual_corr_with_lcore": residual_corr_with_lcore,
        "baseline_mean": float(np.mean(baseline)),
        "lcore_energy_l2": float(np.sqrt(np.mean(lcore ** 2))),
        "residual_energy_l2": float(np.sqrt(np.mean(residual ** 2))),
        "dominant_peak": strongest_peak,
        "peaks": peak_df.to_dict(orient="records"),
    }
    return summary


def summary_to_text(summary: Dict[str, Any]) -> str:
    lines = [
        "FUZE core summary",
        "=================",
        f"points               : {summary['n_points']}",
        f"x-range              : [{summary['x_min']:.6g}, {summary['x_max']:.6g}]",
        f"peaks                : {summary['n_peaks']}",
        f"baseline degree      : {summary['baseline_degree']}",
        f"fit success          : {summary['fit_success']}",
        f"rmse                 : {summary['rmse']:.6g}",
        f"r2                   : {summary['r2']:.6g}",
        f"aic / bic            : {summary['aic']:.6g} / {summary['bic']:.6g}",
        f"transfer_supported   : {summary['transfer_supported']}",
        f"peak_alignment_ok    : {summary['peak_alignment_ok']}",
        f"residual_small       : {summary['residual_small']}",
        f"residual_corr_lcore  : {summary['residual_corr_with_lcore']:.6g}",
        "",
        "Peaks:",
    ]
    for peak in summary.get("peaks", []):
        lines.append(
            "  - "
            f"#{peak['peak_index']} center={peak['center']:.6g}, gamma={peak['gamma']:.6g}, "
            f"amp={peak['amplitude']:.6g}, area_proxy={peak['area_proxy']:.6g}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plotting and serialization
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_fit_report(
    source_info: Dict[str, Any],
    df_xy: pd.DataFrame,
    fits: List[CandidateFit],
    best_fit: CandidateFit,
) -> Dict[str, Any]:
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    fit_y = model_value(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    baseline, lcore, _components = evaluate_components(
        x,
        best_fit.params,
        best_fit.n_peaks,
        best_fit.baseline_degree,
    )
    residual = y - fit_y

    return {
        "source": source_info,
        "n_points": int(len(df_xy)),
        "best_model": asdict(best_fit),
        "top_models": [asdict(item) for item in fits[: min(10, len(fits))]],
        "x_stats": {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        },
        "y_stats": {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        },
        "diagnostics": {
            "fit_y_mean": float(np.mean(fit_y)),
            "baseline_mean": float(np.mean(baseline)),
            "lcore_mean": float(np.mean(lcore)),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
        },
    }


def plot_fit(df_xy: pd.DataFrame, best_fit: CandidateFit, path: Path, title: str) -> None:
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    fit_y = model_value(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    baseline, _lcore, components = evaluate_components(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="data", linewidth=1.8)
    plt.plot(x, fit_y, label="best fit", linewidth=2.0)
    plt.plot(x, baseline, label="baseline", linewidth=1.5, linestyle="--")
    for idx, comp in enumerate(components, start=1):
        plt.plot(x, baseline + comp, label=f"peak {idx}", alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_core(df_xy: pd.DataFrame, best_fit: CandidateFit, path: Path, title: str) -> None:
    x = df_xy["x"].to_numpy(dtype=float)
    y = df_xy["y"].to_numpy(dtype=float)
    fit_y = model_value(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    baseline, lcore, _components = evaluate_components(x, best_fit.params, best_fit.n_peaks, best_fit.baseline_degree)
    residual = y - fit_y

    plt.figure(figsize=(10, 7))
    plt.plot(x, y, label="data", linewidth=1.7)
    plt.plot(x, fit_y, label="fit", linewidth=2.0)
    plt.plot(x, lcore, label="lcore", linewidth=1.5)
    plt.plot(x, residual, label="residual", linewidth=1.2)
    plt.plot(x, baseline, label="baseline", linewidth=1.2, linestyle="--")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def resolve_input(args: argparse.Namespace, output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if args.file:
        file_path = Path(args.file).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Soubor neexistuje: {file_path}")
        df = load_local_table(file_path, mat_key=args.mat_key)
        source_info = {
            "mode": "local_file",
            "path": str(file_path),
            "filename": file_path.name,
        }
        return df, source_info

    if args.doi:
        meta = fetch_dataset_metadata(args.doi)
        files = extract_files(meta)
        candidate = choose_candidate_file(files, file_name_hint=args.file_name)
        write_json(output_dir / "dataset_files.json", {"files": files, "chosen": candidate})
        payload = download_file_bytes(int(candidate["id"]))
        raw_name = candidate.get("filename") or "raw_download.bin"
        (output_dir / raw_name).write_bytes(payload)
        df = load_table_from_bytes(payload, filename=str(raw_name), mat_key=args.mat_key)
        source_info = {
            "mode": "doi",
            "doi": args.doi,
            "filename": raw_name,
            "chosen_file": candidate,
        }
        return df, source_info

    raise RuntimeError("Zadej buď --file, nebo --doi.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FUZE canonical runner")
    parser.add_argument("--file", help="Lokální vstupní datový soubor (csv/tsv/txt/mat).")
    parser.add_argument("--doi", help="DOI veřejného Harvard Dataverse datasetu.")
    parser.add_argument("--file-name", help="Hint pro výběr souboru v datasetu.")
    parser.add_argument(
        "--mat-key",
        help="Top-level MATLAB proměnná / sample key pro .mat soubory.",
    )
    parser.add_argument("--xcol", help="Název x sloupce.")
    parser.add_argument("--ycol", help="Název y sloupce.")
    parser.add_argument("--xmin", type=float, help="Levý ořez x-range.")
    parser.add_argument("--xmax", type=float, help="Pravý ořez x-range.")
    parser.add_argument("--output-dir", default="fuze_out", help="Výstupní adresář.")
    parser.add_argument("--max-peaks", type=int, default=3, help="Max počet Lorentz peaků v searchi.")
    parser.add_argument("--max-baseline-degree", type=int, default=2, help="Max stupeň baseline polynomu.")
    parser.add_argument(
        "--peak-prominence-fraction",
        type=float,
        default=0.08,
        help="Frakce y-span pro detekci seed peaků.",
    )
    parser.add_argument("--title", default="FUZE fit", help="Titulek grafů.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.file) == bool(args.doi):
        parser.error("Zadej právě jednu variantu vstupu: --file nebo --doi.")

    if args.max_peaks < 1:
        parser.error("--max-peaks musí být >= 1")
    if args.max_baseline_degree < 0:
        parser.error("--max-baseline-degree musí být >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df, source_info = resolve_input(args, output_dir)
        df_xy = prepare_xy_dataframe(raw_df, args.xcol, args.ycol, args.xmin, args.xmax)
        df_xy.to_csv(output_dir / "clean_spectrum.csv", index=False)

        x = df_xy["x"].to_numpy(dtype=float)
        y = df_xy["y"].to_numpy(dtype=float)
        fits = rank_models(
            x,
            y,
            max_peaks=args.max_peaks,
            max_baseline_degree=args.max_baseline_degree,
            prominence_fraction=args.peak_prominence_fraction,
        )
        if not fits:
            raise RuntimeError("Nepodařilo se vytvořit žádný kandidátní model.")

        best_fit = fits[0]
        report = make_fit_report(source_info, df_xy, fits, best_fit)
        summary = compute_core_summary(df_xy, best_fit)
        peaks_df = peak_table(best_fit, x)

        write_json(output_dir / "fit_report.json", report)
        write_json(output_dir / "core_summary.json", summary)
        (output_dir / "core_summary.txt").write_text(summary_to_text(summary), encoding="utf-8")
        peaks_df.to_csv(output_dir / "core_table.csv", index=False)

        plot_fit(df_xy, best_fit, output_dir / "fit_plot.png", args.title)
        plot_core(df_xy, best_fit, output_dir / "core_real_plot.png", f"{args.title} — core")

        print(summary_to_text(summary))
        print(f"Outputs saved to: {output_dir.resolve()}")
        return 0
    except Exception as exc:
        print(f"FUZE failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
