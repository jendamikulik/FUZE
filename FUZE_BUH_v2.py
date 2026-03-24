#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
FUZE_BUH_v2.py
--------------
Robustní veřejný Dataverse lovec pro Stage A (rezonanční preflight).

Co dělá:
1) stáhne metadata veřejného datasetu z Dataverse podle DOI,
2) vypíše a uloží index souborů,
3) zkusí vybrat tabulkový kandidát,
4) stáhne raw data přes API access endpoint,
5) načte tabulku (CSV/TSV/TXT/XLSX/XLS/ZIP s tabulkou uvnitř),
6) odhadne x/y sloupce nebo vezme zadané,
7) vyčistí a exportuje clean_spectrum.csv,
8) fitne single i double Lorentzian,
9) uloží graf dat, fitů a residuí + JSON report.

Příklad:
    python FUZE_BUH_v2.py --doi doi:10.7910/DVN/XCS15A
    python FUZE_BUH_v2.py --doi doi:10.7910/DVN/XCS15A --list-only
    python FUZE_BUH_v2.py --doi doi:10.7910/DVN/XCS15A --file-name spectrum --xcol frequency --ycol counts
"""

import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit

DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
USER_AGENT = "FUZE-BUH/2.0 (+public-dataverse-preflight)"
TIMEOUT = 90

TABULAR_EXTS = {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"}
TABLE_MIME_HINTS = (
    "csv",
    "tab-separated-values",
    "excel",
    "spreadsheet",
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats",
    "application/vnd.ms-excel",
)
PREFERRED_NAME_HINTS = (
    "spectrum", "spectra", "freq", "frequency", "counts", "response", "scan", "reson", "cavity", "mode"
)

# ----------------------------
# Dataverse helpers
# ----------------------------

def fetch_dataset_metadata(doi: str, timeout: int = TIMEOUT) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    params = {"persistentId": doi}
    r = requests.get(DATAVERSE_META_URL, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    payload = r.json()
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


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _score_file(item: Dict[str, Any], file_name_hint: Optional[str] = None) -> Tuple[int, int, int]:
    name = str(item.get("filename") or "").lower()
    ctype = str(item.get("contentType") or "").lower()
    ext = Path(name).suffix.lower()

    tab_score = 2 if item.get("tabular") else 0
    tableish = int(ext in TABULAR_EXTS or any(h in ctype for h in TABLE_MIME_HINTS))
    hint_score = 0
    if file_name_hint:
        hint_score += 8 if file_name_hint.lower() in name else 0
    hint_score += sum(1 for h in PREFERRED_NAME_HINTS if h in name)
    size_penalty = -int((item.get("filesize") or 0) / 50_000_000)  # jemná penalizace za obří soubory
    return (tab_score + tableish + hint_score, size_penalty, 0)


def choose_candidate_file(files: Sequence[Dict[str, Any]], file_name_hint: Optional[str] = None) -> Dict[str, Any]:
    if not files:
        raise RuntimeError("Dataset neobsahuje žádné soubory.")
    ranked = sorted(files, key=lambda x: _score_file(x, file_name_hint=file_name_hint), reverse=True)
    best = ranked[0]
    name = str(best.get("filename") or "")
    ctype = str(best.get("contentType") or "")
    if not (best.get("tabular") or Path(name).suffix.lower() in TABULAR_EXTS or any(h in ctype.lower() for h in TABLE_MIME_HINTS)):
        raise RuntimeError(
            "Nepodařilo se najít rozumný tabulkový kandidát. Použij --list-only nebo --file-name."
        )
    return best


def download_file_bytes(file_id: Any, timeout: int = TIMEOUT) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    url = DATAVERSE_FILE_URL.format(file_id=file_id)
    r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    r.raise_for_status()
    content_type = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type and b"<html" in r.content[:500].lower():
        raise RuntimeError(
            f"Stažený obsah vypadá jako HTML ({content_type}), ne raw data. Endpoint nebo file_id jsou špatně."
        )
    return r.content


# ----------------------------
# Table loading
# ----------------------------

def _try_csv(buf: bytes, seps: Iterable[str]) -> Optional[pd.DataFrame]:
    for sep in seps:
        try:
            df = pd.read_csv(io.BytesIO(buf), sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def _load_from_zip(buf: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        ranked = sorted(
            members,
            key=lambda n: (
                Path(n).suffix.lower() in TABULAR_EXTS,
                sum(1 for h in PREFERRED_NAME_HINTS if h in n.lower()),
                -len(n),
            ),
            reverse=True,
        )
        for member in ranked:
            data = zf.read(member)
            try:
                return load_table_from_bytes(data, member)
            except Exception:
                continue
    raise RuntimeError("V ZIPu se nepodařilo najít čitelnou tabulku.")


def load_table_from_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return _load_from_zip(raw)
    if suffix in {".csv", ".txt", ".dat"}:
        df = _try_csv(raw, [",", "\t", ";", r"\s+"])
        if df is not None:
            return df
    if suffix == ".tsv":
        df = _try_csv(raw, ["\t", ",", ";", r"\s+"])
        if df is not None:
            return df
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))

    # content-type fallback by trial
    df = _try_csv(raw, [",", "\t", ";", r"\s+"])
    if df is not None:
        return df
    try:
        return pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        raise RuntimeError(f"Nepodařilo se načíst tabulku ze souboru {filename}: {e}") from e


# ----------------------------
# Cleaning / column choice
# ----------------------------

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9eE+\-\.]+", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_numeric_series(df: pd.DataFrame, min_non_nan_ratio: float = 0.7) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        numeric = _coerce_numeric_series(df[col])
        ratio = float(np.mean(np.isfinite(numeric.to_numpy()))) if len(numeric) else 0.0
        if ratio >= min_non_nan_ratio:
            out[str(col)] = numeric
    if out.shape[1] < 2:
        raise RuntimeError("Po numerickém čištění nezůstaly aspoň dva použitelné sloupce.")
    return out


def guess_xy_columns(df: pd.DataFrame, xcol: Optional[str] = None, ycol: Optional[str] = None) -> Tuple[str, str]:
    cols = list(df.columns)
    if xcol is not None and ycol is not None:
        if xcol not in cols or ycol not in cols:
            raise RuntimeError(f"Zadané sloupce nejsou v tabulce: {xcol}, {ycol}")
        return xcol, ycol

    lower = {c: c.lower() for c in cols}
    x_candidates = []
    y_candidates = []
    for c in cols:
        lc = lower[c]
        x_score = 0
        y_score = 0
        if any(k in lc for k in ["freq", "frequency", "omega", "wavelength", "energy", "time", "field", "detuning", "x"]):
            x_score += 4
        if any(k in lc for k in ["counts", "intensity", "amp", "signal", "response", "power", "y"]):
            y_score += 4
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size >= 5:
            unique_ratio = np.unique(finite).size / finite.size
            x_score += int(unique_ratio > 0.85)
            y_score += int(np.nanstd(finite) > 0)
        x_candidates.append((x_score, c))
        y_candidates.append((y_score, c))

    x_candidates.sort(reverse=True)
    y_candidates.sort(reverse=True)
    best_x = xcol or x_candidates[0][1]
    best_y = ycol or next((c for _, c in y_candidates if c != best_x), cols[1 if cols[0] == best_x else 0])
    if best_x == best_y:
        raise RuntimeError("Nepodařilo se odhadnout odlišné x/y sloupce.")
    return best_x, best_y


# ----------------------------
# Models / fitting
# ----------------------------

def lorentz1(x: np.ndarray, y0: float, A: float, x0: float, gamma: float) -> np.ndarray:
    g2 = np.maximum(gamma, 1e-12) ** 2
    return y0 + A * g2 / ((x - x0) ** 2 + g2)


def lorentz2(x: np.ndarray, y0: float, A1: float, x1: float, g1: float, A2: float, x2: float, g2: float) -> np.ndarray:
    return lorentz1(x, y0, A1, x1, g1) + lorentz1(x, 0.0, A2, x2, g2)


def rss(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


def aic(n: int, k: int, rss_value: float) -> float:
    rss_value = max(rss_value, 1e-30)
    return float(n * np.log(rss_value / n) + 2 * k)


def bic(n: int, k: int, rss_value: float) -> float:
    rss_value = max(rss_value, 1e-30)
    return float(n * np.log(rss_value / n) + k * np.log(n))


def _single_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    y0 = float(np.median(np.r_[y[: max(3, len(y)//10)], y[-max(3, len(y)//10):]]))
    idx = int(np.argmax(y))
    A = float(np.max(y) - y0)
    x0 = float(x[idx])
    x_span = float(np.max(x) - np.min(x))
    gamma = max(x_span / 20.0, 1e-6)
    return y0, A, x0, gamma


def _double_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    y0, A, x0, gamma = _single_guess(x, y)
    y_centered = y - y0
    order = np.argsort(y_centered)[::-1]
    x1 = float(x[order[0]])
    x2 = float(x[order[min(1, len(order)-1)]])
    if abs(x2 - x1) < 1e-12:
        x_span = float(np.max(x) - np.min(x))
        x2 = x1 + 0.15 * x_span
    A1 = 0.6 * A
    A2 = 0.4 * A
    g1 = max(gamma, 1e-6)
    g2 = max(1.5 * gamma, 1e-6)
    return y0, A1, x1, g1, A2, x2, g2


def fit_single_lorentz(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p0 = _single_guess(x, y)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_span = float(np.max(y) - np.min(y))
    bounds = (
        [np.min(y) - y_span, -10 * abs(y_span) - 1, x_min, 1e-12],
        [np.max(y) + y_span, 10 * abs(y_span) + 1, x_max, max(x_max - x_min, 1e-9)],
    )
    popt, pcov = curve_fit(lorentz1, x, y, p0=p0, bounds=bounds, maxfev=200000)
    yhat = lorentz1(x, *popt)
    return {
        "params": {"y0": float(popt[0]), "A": float(popt[1]), "x0": float(popt[2]), "gamma": float(popt[3])},
        "yhat": yhat,
        "rss": rss(y, yhat),
        "cov": np.asarray(pcov).tolist(),
        "n_params": 4,
    }


def fit_double_lorentz(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p0 = _double_guess(x, y)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_span = float(np.max(y) - np.min(y))
    bounds = (
        [np.min(y) - y_span, -10 * abs(y_span) - 1, x_min, 1e-12, -10 * abs(y_span) - 1, x_min, 1e-12],
        [np.max(y) + y_span, 10 * abs(y_span) + 1, x_max, max(x_max - x_min, 1e-9), 10 * abs(y_span) + 1, x_max, max(x_max - x_min, 1e-9)],
    )
    popt, pcov = curve_fit(lorentz2, x, y, p0=p0, bounds=bounds, maxfev=300000)
    yhat = lorentz2(x, *popt)
    return {
        "params": {
            "y0": float(popt[0]), "A1": float(popt[1]), "x1": float(popt[2]), "g1": float(popt[3]),
            "A2": float(popt[4]), "x2": float(popt[5]), "g2": float(popt[6]),
        },
        "yhat": yhat,
        "rss": rss(y, yhat),
        "cov": np.asarray(pcov).tolist(),
        "n_params": 7,
    }


def peak_summary_single(params: Dict[str, float]) -> Dict[str, float]:
    return {
        "center": float(params["x0"]),
        "gamma": float(params["gamma"]),
        "fwhm": float(2.0 * abs(params["gamma"])),
        "amplitude": float(params["A"]),
    }


def peak_summary_double(params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        "peak_1": {
            "center": float(params["x1"]),
            "gamma": float(params["g1"]),
            "fwhm": float(2.0 * abs(params["g1"])),
            "amplitude": float(params["A1"]),
        },
        "peak_2": {
            "center": float(params["x2"]),
            "gamma": float(params["g2"]),
            "fwhm": float(2.0 * abs(params["g2"])),
            "amplitude": float(params["A2"]),
        },
    }


# ----------------------------
# Plotting / report
# ----------------------------

def save_fit_plot(
    x: np.ndarray,
    y: np.ndarray,
    xcol: str,
    ycol: str,
    single: Dict[str, Any],
    double: Dict[str, Any],
    out_path: Path,
) -> None:
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 2000)
    y1_dense = lorentz1(x_dense, *[
        single["params"]["y0"], single["params"]["A"], single["params"]["x0"], single["params"]["gamma"]
    ])
    y2_dense = lorentz2(x_dense, *[
        double["params"]["y0"], double["params"]["A1"], double["params"]["x1"], double["params"]["g1"],
        double["params"]["A2"], double["params"]["x2"], double["params"]["g2"]
    ])

    res1 = y - single["yhat"]
    res2 = y - double["yhat"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[2.2, 1.0], sharex=True)
    ax = axes[0]
    ax.plot(x, y, "o", ms=4, alpha=0.7, label="data")
    ax.plot(x_dense, y1_dense, lw=2, label="single Lorentz")
    ax.plot(x_dense, y2_dense, lw=2, label="double Lorentz")
    ax.set_title("FUZE_BUH v2 — veřejný Dataverse resonance preflight")
    ax.set_ylabel(ycol)
    ax.grid(alpha=0.25)
    ax.legend()

    axr = axes[1]
    axr.axhline(0.0, lw=1)
    axr.plot(x, res1, ".-", ms=3, alpha=0.7, label="single residual")
    axr.plot(x, res2, ".-", ms=3, alpha=0.7, label="double residual")
    axr.set_xlabel(xcol)
    axr.set_ylabel("residual")
    axr.grid(alpha=0.25)
    axr.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_files_index(files: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in files:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", default="doi:10.7910/DVN/XCS15A")
    parser.add_argument("--file-name", default=None, help="část názvu souboru k výběru")
    parser.add_argument("--xcol", default=None)
    parser.add_argument("--ycol", default=None)
    parser.add_argument("--output-dir", default="fuze_buh_v2_out")
    parser.add_argument("--list-only", action="store_true", help="jen vypiš a ulož index souborů")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Stahuju metadata pro {args.doi}")
    meta = fetch_dataset_metadata(args.doi)
    files = extract_files(meta)
    files_index_path = outdir / "files_index.jsonl"
    save_files_index(files, files_index_path)
    print(f"  Nalezeno souborů: {len(files)}")
    print(f"  Index uložen: {files_index_path}")

    if args.list_only:
        print("\nSeznam kandidátů:")
        for item in files:
            print(f"  - id={item['id']}  file={item['filename']}  type={item['contentType']}  tabular={item['tabular']}")
        return 0

    candidate = choose_candidate_file(files, file_name_hint=args.file_name)
    print(f"[2/6] Vybraný soubor: {candidate['filename']} (id={candidate['id']})")

    print("[3/6] Stahuju raw soubor přes api/access/datafile")
    raw = download_file_bytes(candidate["id"])
    local_raw_path = outdir / sanitize_filename(str(candidate["filename"]))
    local_raw_path.write_bytes(raw)
    print(f"  Uloženo: {local_raw_path}")

    print("[4/6] Načítám tabulku")
    df = load_table_from_bytes(raw, str(candidate["filename"]))
    df_num = clean_numeric_series(df)
    xcol, ycol = guess_xy_columns(df_num, xcol=args.xcol, ycol=args.ycol)
    x = pd.to_numeric(df_num[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_num[ycol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 10:
        raise RuntimeError("Po vyčištění zůstalo příliš málo bodů pro fit.")
    clean_csv = outdir / "clean_spectrum.csv"
    pd.DataFrame({xcol: x, ycol: y}).to_csv(clean_csv, index=False)
    print(f"  Sloupce: x='{xcol}', y='{ycol}', body={len(x)}")
    print(f"  Clean CSV: {clean_csv}")

    print("[5/6] Fit: single a double Lorentz")
    single = fit_single_lorentz(x, y)
    double = fit_double_lorentz(x, y)
    n = len(x)
    single["AIC"] = aic(n, single["n_params"], single["rss"])
    single["BIC"] = bic(n, single["n_params"], single["rss"])
    double["AIC"] = aic(n, double["n_params"], double["rss"])
    double["BIC"] = bic(n, double["n_params"], double["rss"])
    preferred = "double_lorentz" if double["BIC"] < single["BIC"] else "single_lorentz"
    print(f"  Preferred by BIC: {preferred}")

    print("[6/6] Ukládám graf a report")
    plot_path = outdir / "fit_plot.png"
    save_fit_plot(x, y, xcol, ycol, single, double, plot_path)

    report = {
        "doi": args.doi,
        "file": candidate,
        "xcol": xcol,
        "ycol": ycol,
        "n_points": int(n),
        "preferred_model_by_BIC": preferred,
        "single_lorentz": {
            "params": single["params"],
            "peak_summary": peak_summary_single(single["params"]),
            "rss": single["rss"],
            "AIC": single["AIC"],
            "BIC": single["BIC"],
        },
        "double_lorentz": {
            "params": double["params"],
            "peak_summary": peak_summary_double(double["params"]),
            "rss": double["rss"],
            "AIC": double["AIC"],
            "BIC": double["BIC"],
        },
        "files_index_path": str(files_index_path),
        "raw_file_path": str(local_raw_path),
        "clean_csv_path": str(clean_csv),
        "plot_path": str(plot_path),
        "notes": [
            "Toto je Stage A resonance preflight, ne plný RTIM fit.",
            "Double model dává smysl jen pokud opravdu vyhraje na BIC a residual plot není artefakt.",
        ],
    }
    report_path = outdir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Graf:   {plot_path}")
    print(f"  Report: {report_path}")
    print("Hotovo.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPřerušeno uživatelem.", file=sys.stderr)
        raise
