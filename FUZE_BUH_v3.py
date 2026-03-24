#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fuze — veřejný Dataverse fit
--------------------------------------

Co to dělá:
1) stáhne metadata veřejného datasetu z Harvard Dataverse podle DOI,
2) najde tabulkový soubor,
3) stáhne ho,
4) zkusí odhadnout x/y sloupce,
5) fitne single a double Lorentzian,
6) porovná modely přes AIC/BIC,
7) uloží graf a JSON report.

Výchozí dataset:
doi:10.7910/DVN/XCS15A

Příklady:
    python fuze_buh.py

    python fuze_buh.py \
        --file-name spectrum \
        --xcol frequency \
        --ycol counts

    python fuze_buh.py \
        --doi doi:10.7910/DVN/XCS15A \
        --output-dir out_cqed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit

import h5py
from scipy.io import loadmat
from scipy.signal import find_peaks

# ----------------------------
# Dataverse helpers
# ----------------------------

DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

#https://dataverse.harvard.edu/file.xhtml?fileId=8241448&version=1.0


def _collect_h5_datasets(h5obj, prefix=""):
    out = {}
    for key in h5obj.keys():
        item = h5obj[key]
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            try:
                out[name] = np.array(item)
            except Exception:
                pass
        elif isinstance(item, h5py.Group):
            out.update(_collect_h5_datasets(item, prefix=name))
    return out


def _flatten_numeric_array(arr):
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    return arr


def _score_key(name: str, patterns):
    lower = str(name).lower()
    score = 0
    for p in patterns:
        if p in lower:
            score += 10
    return score


def _mat_dict_to_dataframe(mat_dict, filename="<mat>"):
    """
    Zkusí z MATLAB dictu vytáhnout smysluplný DataFrame.
    Priorita:
    1) matice tvaru (n,2) nebo (2,n)
    2) dvojice vektorů stejné délky s rozumnými názvy
    3) první dvě numerické vektory stejné délky
    """
    candidates = {}
    for k, v in mat_dict.items():
        if str(k).startswith("__"):
            continue
        arr = _flatten_numeric_array(v)
        if arr is None:
            continue
        candidates[k] = arr

    if not candidates:
        raise RuntimeError(
            f"Soubor '{filename}' neobsahuje čitelné numerické MATLAB proměnné."
        )

    # 1) Jedna 2D matice s dvěma sloupci / řádky
    matrix_candidates = []
    for k, arr in candidates.items():
        if arr.ndim == 2 and min(arr.shape) == 2 and max(arr.shape) >= 10:
            matrix_candidates.append((k, arr))

    if matrix_candidates:
        # vezmi největší
        k, arr = sorted(matrix_candidates, key=lambda kv: kv[1].size, reverse=True)[0]
        if arr.shape[1] == 2:
            df = pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
            return df
        elif arr.shape[0] == 2:
            df = pd.DataFrame({"x": arr[0, :], "y": arr[1, :]})
            return df

    # 2) Dvojice vektorů stejné délky, preferuj názvy
    vector_candidates = {}
    for k, arr in candidates.items():
        if arr.ndim == 1 and len(arr) >= 10:
            vector_candidates[k] = arr

    if len(vector_candidates) >= 2:
        x_patterns = ["x", "freq", "frequency", "omega", "detuning", "wavelength", "energy", "time"]
        y_patterns = ["y", "count", "counts", "signal", "intensity", "trans", "transmission", "reflect", "power"]

        keys = list(vector_candidates.keys())

        best_pair = None
        best_score = None

        for kx in keys:
            for ky in keys:
                if kx == ky:
                    continue
                xarr = vector_candidates[kx]
                yarr = vector_candidates[ky]
                if len(xarr) != len(yarr):
                    continue

                score = (
                    _score_key(kx, x_patterns),
                    _score_key(ky, y_patterns),
                    len(xarr),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (kx, ky)

        if best_pair is not None:
            kx, ky = best_pair
            return pd.DataFrame({kx: vector_candidates[kx], ky: vector_candidates[ky]})

        # 3) fallback: první dvě vektory stejné délky
        vec_items = list(vector_candidates.items())
        for i in range(len(vec_items)):
            for j in range(i + 1, len(vec_items)):
                kx, xarr = vec_items[i]
                ky, yarr = vec_items[j]
                if len(xarr) == len(yarr):
                    return pd.DataFrame({kx: xarr, ky: yarr})

    # 4) Když nic, vypiš klíče
    shapes = {k: tuple(np.asarray(v).shape) for k, v in candidates.items()}
    raise RuntimeError(
        f"Nepodařilo se automaticky rozpoznat x/y data v '{filename}'. "
        f"Dostupné MATLAB proměnné a tvary: {shapes}"
    )


def load_mat_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """
    Podpora pro klasické MATLAB .mat i HDF5-based v7.3.
    """
    # Klasický .mat
    try:
        bio = io.BytesIO(data)
        mat = loadmat(bio, squeeze_me=True, struct_as_record=False)
        return _mat_dict_to_dataframe(mat, filename=filename)
    except NotImplementedError:
        # typicky v7.3, zkusíme h5py
        pass
    except Exception:
        # ještě zkusíme h5py fallback
        pass

    if h5py is None:
        raise RuntimeError(
            f"Soubor '{filename}' vypadá jako MATLAB v7.3/HDF5, ale chybí balík h5py."
        )

    try:
        bio = io.BytesIO(data)
        with h5py.File(bio, "r") as f:
            h5dict = _collect_h5_datasets(f)
        return _mat_dict_to_dataframe(h5dict, filename=filename)
    except Exception as e:
        raise RuntimeError(
            f"Nepodařilo se načíst MATLAB soubor '{filename}' ani přes scipy.io.loadmat, ani přes h5py. "
            f"Detail: {e}"
        )


def fetch_dataset_metadata(doi: str, timeout: int = 60) -> Dict[str, Any]:
    params = {"persistentId": doi}
    r = requests.get(DATAVERSE_META_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API error: {payload}")
    return payload["data"]


def extract_files(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Vytáhne seznam souborů z dataset metadata.
    """
    latest = meta.get("latestVersion", {})
    files = latest.get("files", [])
    out = []

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
    """
    Vybere kandidátní soubor:
    - pokud je zadaný hint, zkusí filename obsahující hint
    - jinak preferuje csv/tsv/txt/tab/tabular
    """
    if not files:
        raise RuntimeError("Dataset neobsahuje žádné soubory.")

    if file_name_hint:
        hint = file_name_hint.lower()
        hinted = [f for f in files if hint in (f.get("filename") or "").lower()]
        if hinted:
            return hinted[0]

    def score(f: Dict[str, Any]) -> Tuple[int, int]:
        name = (f.get("filename") or "").lower()
        ctype = (f.get("contentType") or "").lower()

        score_val = 0
        if f.get("tabular"):
            score_val += 100
        if name.endswith(".csv"):
            score_val += 50
        if name.endswith(".tsv"):
            score_val += 45
        if name.endswith(".txt"):
            score_val += 30
        if name.endswith(".dat"):
            score_val += 20

        if "csv" in ctype:
            score_val += 40
        if "tab-separated" in ctype:
            score_val += 35
        if "text/plain" in ctype:
            score_val += 15

        # menší soubory trochu preferujeme, aby to bylo svižnější
        filesize = int(f.get("filesize") or 10**12)
        return (score_val, -filesize)

    ranked = sorted(files, key=score, reverse=True)
    best = ranked[0]
    if score(best)[0] <= 0:
        raise RuntimeError(
            "Nenašel jsem rozumný tabulkový soubor. "
            "Použij --file-name a vyber ho ručně."
        )
    return best


def download_file_bytes(file_id: int, timeout: int = 120) -> bytes:
    url = DATAVERSE_FILE_URL.format(file_id=file_id)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


# ----------------------------
# Tabular parsing
# ----------------------------

def sniff_delimiter(sample: str, fallback: str = ",") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return fallback


def load_table_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """
    Zkusí načíst CSV/TSV/TXT i MATLAB .mat robustně.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".mat":
        return load_mat_from_bytes(data, filename)

    text = data.decode("utf-8", errors="replace")
    sample = text[:5000]

    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = sniff_delimiter(sample, fallback=",")
    else:
        sep = sniff_delimiter(sample, fallback=",")

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        if df.shape[1] >= 2:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            return df
    except Exception:
        pass

    raise RuntimeError(
        f"Nepodařilo se načíst tabulku ze souboru '{filename}'. "
        f"Zkus vybrat jiný soubor pomocí --file-name."
    )


def clean_numeric_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = (
                out[c]
                .astype(str)
                .str.replace("\u2212", "-", regex=False)  # unicode minus
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\dEe\+\-\.]", "", regex=True)
            )
        out[c] = pd.to_numeric(out[c])
    return out


def guess_xy_columns(
    df: pd.DataFrame,
    xcol: Optional[str] = None,
    ycol: Optional[str] = None,
) -> Tuple[str, str]:
    cols = list(df.columns)

    if xcol and ycol:
        if xcol not in df.columns or ycol not in df.columns:
            raise RuntimeError(
                f"Zadané sloupce xcol='{xcol}' nebo ycol='{ycol}' v datech nejsou."
            )
        return xcol, ycol

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise RuntimeError(
            "Nenašel jsem aspoň dva numerické sloupce. "
            "Zkus ručně zadat --xcol a --ycol."
        )

    x_patterns = [
        "frequency", "freq", "omega", "detuning", "wavelength",
        "energy", "time", "x", "scan", "voltage",
    ]
    y_patterns = [
        "count", "counts", "signal", "intensity", "trans",
        "transmission", "reflect", "reflection", "power", "y", "phot",
    ]

    def score_name(name: str, patterns: List[str]) -> int:
        lower = str(name).lower()
        return sum(10 for p in patterns if p in lower)

    x_scores = {c: score_name(c, x_patterns) for c in numeric_cols}
    y_scores = {c: score_name(c, y_patterns) for c in numeric_cols}

    # x: preferuj jméno nebo jinak první numerický s nejvíc unikátními hodnotami
    x_candidate = max(
        numeric_cols,
        key=lambda c: (x_scores[c], df[c].nunique(dropna=True))
    )

    # y: nechť není x, preferuj jméno nebo jinak druhý numerický
    y_candidates = [c for c in numeric_cols if c != x_candidate]
    y_candidate = max(
        y_candidates,
        key=lambda c: (y_scores[c], df[c].notna().sum())
    )

    return x_candidate, y_candidate


# ----------------------------
# Lorentz models
# ----------------------------

def lorentz1(x: np.ndarray, y0: float, A: float, x0: float, gamma: float) -> np.ndarray:
    return y0 + A * gamma**2 / ((x - x0)**2 + gamma**2)


def lorentz2(
    x: np.ndarray,
    y0: float,
    A1: float, x1: float, g1: float,
    A2: float, x2: float, g2: float
) -> np.ndarray:
    return (
        y0
        + A1 * g1**2 / ((x - x1)**2 + g1**2)
        + A2 * g2**2 / ((x - x2)**2 + g2**2)
    )


def rss(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


def aic(n: int, k: int, rss_val: float) -> float:
    rss_safe = max(rss_val, 1e-15)
    return n * np.log(rss_safe / n) + 2 * k


def bic(n: int, k: int, rss_val: float) -> float:
    rss_safe = max(rss_val, 1e-15)
    return n * np.log(rss_safe / n) + k * np.log(n)


def initial_guess_single(x: np.ndarray, y: np.ndarray) -> List[float]:
    idx = int(np.argmax(y))
    y0 = float(np.median(y))
    A = float(np.max(y) - y0)
    x0 = float(x[idx])
    gamma = float(max((np.max(x) - np.min(x)) / 20.0, 1e-6))
    return [y0, A, x0, gamma]


def initial_guess_double(x: np.ndarray, y: np.ndarray) -> List[float]:
    y0 = float(np.median(y))
    x_sorted = np.argsort(x)
    xs = x[x_sorted]
    ys = y[x_sorted]

    # najdi dva vrcholy hrubě
    peak_indices = np.argpartition(ys, -2)[-2:]
    peak_indices = peak_indices[np.argsort(xs[peak_indices])]

    if len(peak_indices) < 2:
        peak_indices = np.array([len(xs)//3, 2*len(xs)//3])

    x1, x2 = float(xs[peak_indices[0]]), float(xs[peak_indices[1]])
    A1 = float(max(ys[peak_indices[0]] - y0, 1e-6))
    A2 = float(max(ys[peak_indices[1]] - y0, 1e-6))
    g = float(max((np.max(x) - np.min(x)) / 30.0, 1e-6))

    return [y0, A1, x1, g, A2, x2, g]


def fit_single_lorentz(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p0 = initial_guess_single(x, y)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_span = float(np.max(y) - np.min(y))
    bounds = (
        [np.min(y) - y_span, -10 * abs(y_span) - 1, x_min, 1e-9],
        [np.max(y) + y_span,  10 * abs(y_span) + 1, x_max, (x_max - x_min)]
    )

    popt, pcov = curve_fit(
        lorentz1,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    yhat = lorentz1(x, *popt)
    return {
        "params": {
            "y0": float(popt[0]),
            "A": float(popt[1]),
            "x0": float(popt[2]),
            "gamma": float(popt[3]),
        },
        "yhat": yhat,
        "rss": rss(y, yhat),
        "cov": pcov.tolist(),
        "n_params": 4,
    }


def fit_double_lorentz(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p0 = initial_guess_double(x, y)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_span = float(np.max(y) - np.min(y))
    bounds = (
        [np.min(y) - y_span, -10 * abs(y_span) - 1, x_min, 1e-9,
         -10 * abs(y_span) - 1, x_min, 1e-9],
        [np.max(y) + y_span,  10 * abs(y_span) + 1, x_max, (x_max - x_min),
          10 * abs(y_span) + 1, x_max, (x_max - x_min)]
    )

    popt, pcov = curve_fit(
        lorentz2,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )
    yhat = lorentz2(x, *popt)
    return {
        "params": {
            "y0": float(popt[0]),
            "A1": float(popt[1]),
            "x1": float(popt[2]),
            "g1": float(popt[3]),
            "A2": float(popt[4]),
            "x2": float(popt[5]),
            "g2": float(popt[6]),
        },
        "yhat": yhat,
        "rss": rss(y, yhat),
        "cov": pcov.tolist(),
        "n_params": 7,
    }


# --- FIX ---
def _matobj_to_python(obj, depth=0, max_depth=6):
    """
    Rekurzivně převede MATLAB object/struct/cell na Python dict/list/ndarray.
    Funguje pro scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False).
    """
    if depth > max_depth:
        return obj

    # MATLAB struct jako objekt s _fieldnames
    if hasattr(obj, "_fieldnames"):
        out = {}
        for name in obj._fieldnames:
            try:
                out[name] = _matobj_to_python(getattr(obj, name), depth + 1, max_depth)
            except Exception:
                pass
        return out

    # object ndarray => cell array / zabalené objekty
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 0:
            try:
                return _matobj_to_python(obj.item(), depth + 1, max_depth)
            except Exception:
                return obj
        return [_matobj_to_python(x, depth + 1, max_depth) for x in obj.flat]

    return obj


def _extract_numeric_candidates(obj, prefix="root"):
    """
    Vytáhne numerické kandidáty z libovolně vnořené Python struktury.
    Vrací dict name -> ndarray/list.
    """
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = _extract_numeric_candidates(v, f"{prefix}/{k}")
            out.update(sub)
        return out

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            sub = _extract_numeric_candidates(v, f"{prefix}[{i}]")
            out.update(sub)
        return out

    arr = np.asarray(obj)
    if np.issubdtype(arr.dtype, np.number):
        arr = np.squeeze(arr)
        if arr.ndim >= 1 and arr.size >= 2:
            out[prefix] = arr
    return out


def _mat_dict_to_dataframe(mat_dict, filename="<mat>"):
    """
    Zkusí z MATLAB dictu vytáhnout DataFrame.
    Priorita:
    1) numerická matice (n,2) nebo (2,n)
    2) dvojice vektorů stejné délky
    3) fallback s podrobným výpisem kandidátů
    """
    # 1) převod MATLAB objektů na Python
    converted = {}
    for k, v in mat_dict.items():
        if str(k).startswith("__"):
            continue
        converted[k] = _matobj_to_python(v)

    # 2) vytáhni numerické kandidáty
    candidates = {}
    for k, v in converted.items():
        candidates.update(_extract_numeric_candidates(v, prefix=k))

    if not candidates:
        raise RuntimeError(
            f"Soubor '{filename}' neobsahuje žádné čitelné numerické kandidáty."
        )

    # 3) preferuj matici (n,2) nebo (2,n)
    matrix_candidates = []
    for k, arr in candidates.items():
        arr = np.asarray(arr)
        if arr.ndim == 2 and min(arr.shape) == 2 and max(arr.shape) >= 10:
            matrix_candidates.append((k, arr))

    if matrix_candidates:
        k, arr = sorted(matrix_candidates, key=lambda kv: kv[1].size, reverse=True)[0]
        if arr.shape[1] == 2:
            return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
        if arr.shape[0] == 2:
            return pd.DataFrame({"x": arr[0, :], "y": arr[1, :]})

    # 4) pak zkus dvojici vektorů stejné délky
    vector_candidates = {}
    for k, arr in candidates.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) >= 10:
            vector_candidates[k] = arr

    if len(vector_candidates) >= 2:
        x_patterns = ["x", "freq", "frequency", "omega", "detuning", "wavelength", "energy", "time"]
        y_patterns = ["y", "count", "counts", "signal", "intensity", "trans", "transmission", "reflect", "power"]

        def score_key(name, pats):
            lower = str(name).lower()
            return sum(10 for p in pats if p in lower)

        keys = list(vector_candidates.keys())
        best_pair = None
        best_score = None

        for kx in keys:
            for ky in keys:
                if kx == ky:
                    continue
                xarr = vector_candidates[kx]
                yarr = vector_candidates[ky]
                if len(xarr) != len(yarr):
                    continue

                score = (
                    score_key(kx, x_patterns),
                    score_key(ky, y_patterns),
                    len(xarr),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (kx, ky)

        if best_pair is not None:
            kx, ky = best_pair
            return pd.DataFrame({kx: vector_candidates[kx], ky: vector_candidates[ky]})

    # 5) fallback: ukaž inventář
    shapes = {}
    for k, arr in candidates.items():
        try:
            shapes[k] = tuple(np.asarray(arr).shape)
        except Exception:
            shapes[k] = "<unreadable>"

    raise RuntimeError(
        f"Nepodařilo se automaticky rozpoznat x/y data v '{filename}'. "
        f"Dostupní numeričtí kandidáti: {shapes}"
    )


def load_mat_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """
    Podpora pro klasické MATLAB .mat i HDF5-based v7.3.
    """
    # nejdřív klasický .mat
    try:
        bio = io.BytesIO(data)
        mat = loadmat(bio, squeeze_me=True, struct_as_record=False)
        return _mat_dict_to_dataframe(mat, filename=filename)
    except NotImplementedError:
        # typicky v7.3
        pass
    except Exception as e:
        classic_error = e
    else:
        classic_error = None

    # fallback pro v7.3 / HDF5
    if h5py is None:
        raise RuntimeError(
            f"Soubor '{filename}' nejde přečíst přes scipy.io.loadmat "
            f"a balík h5py není dostupný. Detail: {classic_error}"
        )

    try:
        bio = io.BytesIO(data)
        with h5py.File(bio, "r") as f:
            h5dict = _collect_h5_datasets(f)
        return _mat_dict_to_dataframe(h5dict, filename=filename)
    except Exception as e:
        raise RuntimeError(
            f"Nepodařilo se načíst MATLAB soubor '{filename}'. "
            f"loadmat detail: {classic_error}; h5py detail: {e}"
        )

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


# --- fix ---
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def poly_baseline(x, coeffs):
    y = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        y += c * x**i
    return y


def lorentz_sum(x, peak_params):
    """
    peak_params = [A1, x1, g1, A2, x2, g2, ...]
    """
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(peak_params), 3):
        A = peak_params[i]
        x0 = peak_params[i + 1]
        g = peak_params[i + 2]
        y += A * g**2 / ((x - x0)**2 + g**2)
    return y


def make_multi_lorentz_model(n_peaks, baseline_degree):
    n_base = baseline_degree + 1

    def model(x, *params):
        base_coeffs = params[:n_base]
        peak_params = params[n_base:]
        return poly_baseline(x, base_coeffs) + lorentz_sum(x, peak_params)

    return model


def rss(y, yhat):
    return float(np.sum((y - yhat) ** 2))


def aic(n, k, rss_val):
    rss_safe = max(float(rss_val), 1e-15)
    return n * np.log(rss_safe / n) + 2 * k


def bic(n, k, rss_val):
    rss_safe = max(float(rss_val), 1e-15)
    return n * np.log(rss_safe / n) + k * np.log(n)


def _initial_baseline_guess(x, y, degree):
    if degree == 0:
        return [float(np.median(y))]
    if degree == 1:
        # lineární fit na celé spektrum jako hrubý start
        c1, c0 = np.polyfit(x, y, 1)
        return [float(c0), float(c1)]
    if degree == 2:
        c2, c1, c0 = np.polyfit(x, y, 2)
        return [float(c0), float(c1), float(c2)]
    raise ValueError("baseline_degree musí být 0, 1 nebo 2")


def _peak_candidates(x, y, n_peaks, baseline_guess):
    """
    Najde kandidátní peaky z residualu nad baseline.
    """
    y_base = poly_baseline(x, baseline_guess)
    resid = y - y_base

    # chceme pozitivní lokální maxima
    prominence = max(np.std(resid) * 0.25, (np.max(resid) - np.min(resid)) * 0.03, 1e-12)
    distance = max(3, len(x) // (4 * max(n_peaks, 1)))

    peaks, props = find_peaks(resid, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        # fallback: vezmi globální maxima residualu
        peaks = np.argsort(resid)[-n_peaks:]
        peaks = np.array(sorted(peaks))

    # seřaď podle výšky residualu
    peaks = sorted(peaks, key=lambda idx: resid[idx], reverse=True)

    chosen = []
    used_x = []
    min_sep = max((np.max(x) - np.min(x)) / (8 * max(n_peaks, 1)), 1e-9)

    for idx in peaks:
        xi = x[idx]
        if all(abs(xi - ux) > min_sep for ux in used_x):
            chosen.append(idx)
            used_x.append(xi)
        if len(chosen) >= n_peaks:
            break

    # když jich je málo, doplň globálními kandidáty
    if len(chosen) < n_peaks:
        fallback = np.argsort(resid)[::-1]
        for idx in fallback:
            xi = x[idx]
            if all(abs(xi - ux) > min_sep for ux in used_x):
                chosen.append(idx)
                used_x.append(xi)
            if len(chosen) >= n_peaks:
                break

    chosen = sorted(chosen, key=lambda idx: x[idx])
    return chosen, resid


def initial_guess_multi(x, y, n_peaks, baseline_degree):
    base_guess = _initial_baseline_guess(x, y, baseline_degree)
    peak_idx, resid = _peak_candidates(x, y, n_peaks, base_guess)

    span = float(np.max(x) - np.min(x))
    default_g = max(span / (10 * max(n_peaks, 1)), 1e-6)

    peak_params = []
    for idx in peak_idx:
        A = max(float(resid[idx]), 1e-9)
        x0 = float(x[idx])
        g = default_g
        peak_params.extend([A, x0, g])

    # doplň když je peaků míň
    while len(peak_params) < 3 * n_peaks:
        peak_params.extend([max(np.max(y) - np.median(y), 1e-6), float(np.median(x)), default_g])

    return base_guess + peak_params


def bounds_multi(x, y, n_peaks, baseline_degree):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_span = max(y_max - y_min, 1e-9)
    x_span = max(x_max - x_min, 1e-9)

    # baseline bounds
    lower = []
    upper = []

    # c0
    lower.append(y_min - 5 * y_span)
    upper.append(y_max + 5 * y_span)

    if baseline_degree >= 1:
        lower.append(-10 * y_span / x_span)
        upper.append(+10 * y_span / x_span)

    if baseline_degree >= 2:
        lower.append(-10 * y_span / (x_span**2))
        upper.append(+10 * y_span / (x_span**2))

    # peak bounds
    for _ in range(n_peaks):
        # A, x0, g
        lower.extend([-5 * y_span, x_min, 1e-9])
        upper.extend([+10 * y_span, x_max, x_span])

    return (lower, upper)


def fit_multi_lorentz(x, y, n_peaks=2, baseline_degree=1):
    model = make_multi_lorentz_model(n_peaks, baseline_degree)
    p0 = initial_guess_multi(x, y, n_peaks, baseline_degree)
    bounds = bounds_multi(x, y, n_peaks, baseline_degree)

    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )

    yhat = model(x, *popt)
    n_base = baseline_degree + 1
    base_coeffs = popt[:n_base]
    peak_params = popt[n_base:]
    k = len(popt)

    peaks = []
    for i in range(0, len(peak_params), 3):
        peaks.append(
            {
                "A": float(peak_params[i]),
                "x0": float(peak_params[i + 1]),
                "gamma": float(peak_params[i + 2]),
                "fwhm": float(2 * peak_params[i + 2]),
            }
        )

    return {
        "n_peaks": n_peaks,
        "baseline_degree": baseline_degree,
        "params_raw": [float(v) for v in popt],
        "baseline_coeffs": [float(v) for v in base_coeffs],
        "peaks": peaks,
        "yhat": yhat,
        "rss": rss(y, yhat),
        "cov": pcov.tolist(),
        "n_params": k,
    }


def fit_model_grid(x, y, max_peaks=4, max_baseline_degree=2):
    candidates = []

    for deg in range(max_baseline_degree + 1):
        for n_peaks in range(1, max_peaks + 1):
            try:
                fit = fit_multi_lorentz(x, y, n_peaks=n_peaks, baseline_degree=deg)
                fit["AIC"] = aic(len(x), fit["n_params"], fit["rss"])
                fit["BIC"] = bic(len(x), fit["n_params"], fit["rss"])
                candidates.append(fit)
            except Exception as e:
                candidates.append(
                    {
                        "n_peaks": n_peaks,
                        "baseline_degree": deg,
                        "fit_failed": True,
                        "error": str(e),
                        "BIC": np.inf,
                        "AIC": np.inf,
                    }
                )

    ok = [c for c in candidates if not c.get("fit_failed")]
    if not ok:
        raise RuntimeError("Všechny multi-peak fit pokusy selhaly.")

    best = min(ok, key=lambda c: c["BIC"])
    ok_sorted = sorted(ok, key=lambda c: c["BIC"])
    return best, ok_sorted, candidates

# ----------------------------
# Main pipeline
# ----------------------------

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", default="doi:10.7910/DVN/XCS15A")
    parser.add_argument("--file-name", default="fig2c_data.mat", help="část názvu souboru k výběru")
    parser.add_argument("--xcol", default=None)
    parser.add_argument("--ycol", default=None)
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--xmin", type=float, default=20)
    parser.add_argument("--xmax", type=float, default=60)
    parser.add_argument("--max-peaks", type=int, default=4)
    parser.add_argument("--max-baseline-degree", type=int, default=2)

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Stahuju metadata pro {args.doi}")
    meta = fetch_dataset_metadata(args.doi)
    files = extract_files(meta)

    files_index_path = outdir / "files_index.jsonl"
    with files_index_path.open("w", encoding="utf-8") as f:
        for item in files:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Nalezeno souborů: {len(files)}")
    print(f"  Index uložen: {files_index_path}")

    candidate = choose_candidate_file(files, file_name_hint=args.file_name)
    print(f"[2/6] Vybraný soubor: {candidate['filename']} (id={candidate['id']})")

    print("[3/6] Stahuju soubor")
    raw = download_file_bytes(candidate["id"])
    local_raw_path = outdir / sanitize_filename(candidate["filename"])
    local_raw_path.write_bytes(raw)
    print(f"  Uloženo: {local_raw_path}")








    print("[4/6] Načítám tabulku")
    df = load_table_from_bytes(raw, candidate["filename"])
    df = clean_numeric_series(df)
    xcol, ycol = guess_xy_columns(df, xcol=args.xcol, ycol=args.ycol)
    print(f"  Použité sloupce: x='{xcol}', y='{ycol}'")

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()

    """mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        raise RuntimeError("Po vyčištění zůstalo příliš málo bodů pro fit.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    print(f"[5/6] Fit")"""

    # optional fit window
    if args.xmin is not None:
        mask = x >= args.xmin
        x = x[mask]
        y = y[mask]

    if args.xmax is not None:
        mask = x <= args.xmax
        x = x[mask]
        y = y[mask]

    if len(x) < 20:
        raise RuntimeError("Po oříznutí fit window zůstalo příliš málo bodů.")

    print("[5/6] Multi-model fit")
    best, ranked_models, all_models = fit_model_grid(
        x,
        y,
        max_peaks=args.max_peaks,
        max_baseline_degree=args.max_baseline_degree,
    )

    print(
        f"  Best model by BIC: "
        f"baseline_degree={best['baseline_degree']}, "
        f"n_peaks={best['n_peaks']}, "
        f"BIC={best['BIC']:.3f}"
    )

    single = fit_single_lorentz(x, y)
    double = fit_double_lorentz(x, y)

    n = len(x)
    single["AIC"] = aic(n, single["n_params"], single["rss"])
    single["BIC"] = bic(n, single["n_params"], single["rss"])

    double["AIC"] = aic(n, double["n_params"], double["rss"])
    double["BIC"] = bic(n, double["n_params"], double["rss"])

    preferred = "double_lorentz" if double["BIC"] < single["BIC"] else "single_lorentz"
    print(f"  Preferred by BIC: {preferred}")

    """print("[6/6] Ukládám report a graf")
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1500)

    plt.figure(figsize=(11, 7))
    plt.plot(x, y, "o", ms=3, alpha=0.7, label="data")
    plt.plot(
        x_dense,
        lorentz1(x_dense, *[
            single["params"]["y0"],
            single["params"]["A"],
            single["params"]["x0"],
            single["params"]["gamma"],
        ]),
        lw=2,
        label="single Lorentz",
    )
    plt.plot(
        x_dense,
        lorentz2(x_dense, *[
            double["params"]["y0"],
            double["params"]["A1"],
            double["params"]["x1"],
            double["params"]["g1"],
            double["params"]["A2"],
            double["params"]["x2"],
            double["params"]["g2"],
        ]),
        lw=2,
        label="double Lorentz",
    )
    plt.xlabel(safe_plot_label(xcol))
    plt.ylabel(safe_plot_label(ycol))
    plt.title("fuze — veřejný Dataverse fit")
    plt.legend()
    plt.grid(alpha=0.25)
    plot_path = outdir / "fit_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()"""

    print("[6/6] Ukládám report a graf")
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 2000)

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, "o", ms=3, alpha=0.65, label="data")

    # vykresli top 3 modely
    top_models = ranked_models[:3]
    colors = ["tab:orange", "tab:green", "tab:red"]

    for rank, (model_fit, color) in enumerate(zip(top_models, colors), start=1):
        model = make_multi_lorentz_model(model_fit["n_peaks"], model_fit["baseline_degree"])
        y_dense = model(x_dense, *model_fit["params_raw"])
        label = (
            f"rank {rank}: deg={model_fit['baseline_degree']}, "
            f"peaks={model_fit['n_peaks']}, "
            f"BIC={model_fit['BIC']:.1f}"
        )
        plt.plot(x_dense, y_dense, lw=2, color=color, label=label)

    # baseline + komponenty pro best model
    best_model = make_multi_lorentz_model(best["n_peaks"], best["baseline_degree"])
    n_base = best["baseline_degree"] + 1
    base_coeffs = best["params_raw"][:n_base]
    peak_params = best["params_raw"][n_base:]

    baseline_dense = poly_baseline(x_dense, base_coeffs)
    plt.plot(x_dense, baseline_dense, "--", color="black", alpha=0.8, label="best baseline")

    # jednotlivé komponenty
    for i in range(0, len(peak_params), 3):
        A = peak_params[i]
        x0 = peak_params[i + 1]
        g = peak_params[i + 2]
        comp = baseline_dense + A * g ** 2 / ((x_dense - x0) ** 2 + g ** 2)
        plt.plot(x_dense, comp, ":", lw=1.5, alpha=0.8)

    plt.xlabel(safe_plot_label(xcol))
    plt.ylabel(safe_plot_label(ycol))
    plt.title("fuze — veřejný Dataverse fit")
    plt.legend()
    plt.grid(alpha=0.25)

    plot_path = outdir / "fit_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    report = {
        "doi": args.doi,
        "file": candidate,
        "xcol": xcol,
        "ycol": ycol,
        "fit_window": {
            "xmin": args.xmin,
            "xmax": args.xmax,
        },
        "n_points": int(len(x)),
        "best_model_by_BIC": {
            "baseline_degree": best["baseline_degree"],
            "n_peaks": best["n_peaks"],
            "rss": best["rss"],
            "AIC": best["AIC"],
            "BIC": best["BIC"],
            "baseline_coeffs": best["baseline_coeffs"],
            "peaks": best["peaks"],
        },
        "top_models": [
            {
                "baseline_degree": m["baseline_degree"],
                "n_peaks": m["n_peaks"],
                "rss": m["rss"],
                "AIC": m["AIC"],
                "BIC": m["BIC"],
                "peaks": m["peaks"],
            }
            for m in ranked_models[:10]
        ],
        "files_index_path": str(files_index_path),
        "raw_file_path": str(local_raw_path),
        "plot_path": str(plot_path),
    }

    report_path = outdir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Graf:   {plot_path}")
    print(f"  Report: {report_path}")
    print("Hotovo.")
    return 0


if __name__ == "__main__":
    main()