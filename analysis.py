"""Recalculate descriptive results from the released, processed data.

This module does not fit or claim to reproduce the historical ML models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import load_bundle, resource_path

CLASS_NAMES = {
    "酯类": "Esters", "醇类": "Alcohols", "酸类": "Acids",
    "醛酮类": "Aldehydes and ketones", "其它类": "Others",
    "萜烯类": "Terpenes", "烷烃类": "Alkanes",
}
SAMPLE_NAMES = {
    "WB": "MW", "HK": "BK", "ZZ": "ST", "SZ": "BO",
    "全浆发酵": "WMF", "浸渍后发酵": "PMF", "清汁发酵": "CJF",
}
FEATURE_NAMES = {
    "酒精度": "Alcohol", "还原糖": "Reducing sugar", "总酸": "Total acid",
    "pH": "pH", "总酚": "Total phenolics", "总黄酮": "Total flavonoids",
    "可溶性蛋白": "Soluble protein", "DPPH": "DPPH scavenging",
    "总还原能力": "Reducing power",
}


def eta_squared(values, groups) -> float:
    """Between-group / total sum of squares; undefined for constant data."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("values and groups must be equal-length one-dimensional arrays")
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("At least two finite observations are required")
    total = np.square(values - values.mean()).sum()
    if total == 0:
        raise ValueError("eta-squared is undefined for a constant index")
    between = sum(
        np.sum(groups == group) * (values[groups == group].mean() - values.mean()) ** 2
        for group in np.unique(groups)
    )
    return float(between / total)


def variance_table() -> pd.DataFrame:
    ml = load_bundle("ml")
    frame = ml["W"]
    methods = frame["处理"].to_numpy()
    binary = np.isin(methods, ["WB", "HK"]).astype(int)
    rows = []
    for feature in ml["FEAT"]:
        method_eta = eta_squared(frame[feature], methods)
        rows.append({
            "index": FEATURE_NAMES[feature],
            "eta2_method": method_eta,
            "eta2_operational_binary_group": eta_squared(frame[feature], binary),
            "residual_within_method": 1 - method_eta,
        })
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = ["Mean of nine indices", *frame.iloc[:, 1:].mean().tolist()]
    return frame


def class_summary() -> pd.DataFrame:
    paper = load_bundle("paper")
    records = []
    for experiment, key, samples in [
        ("cultivar", "vol_var", ["HN", "BT", "HT"]),
        ("thermal", "vol_heat", ["WB", "HK", "ZZ", "SZ"]),
    ]:
        frame = paper[key]
        groups = [("Total", frame)] + [
            (CLASS_NAMES[label], subset) for label, subset in frame.groupby("类别")
        ]
        for label, subset in groups:
            for sample in samples:
                mean = subset[f"{sample}_均值"]
                sd = subset[f"{sample}_SD"]
                records.append({
                    "experiment": experiment, "sample": SAMPLE_NAMES.get(sample, sample),
                    "chemical_class": label, "detected_count": int(mean.notna().sum()),
                    "concentration_ug_L_2_octanol_equivalents": float(mean.fillna(0).sum()),
                    "approximate_propagated_sd_zero_covariance": float(np.sqrt(np.square(sd.fillna(0)).sum())),
                })
    return pd.DataFrame(records)


def composite_table() -> pd.DataFrame:
    paper = load_bundle("paper")
    records = []
    for experiment, key, columns, weights in [
        ("cultivar", "z_var", ["Z_酒精度", "Z_总酚", "Z_DPPH", "Z_总还原能力", "Z_感官"], [.10, .24, .16, .33, .17]),
        ("thermal", "z_heat", ["Z_酒精度", "Z_总酚", "Z_DPPH", "Z_总还原能力", "Z_感官"], [.10, .24, .16, .33, .17]),
        ("fermentation", "z_mode", ["Z_总酚", "Z_总还原能力", "Z_氨基酸态氮", "Z_非糖固形物", "Z_感官"], [.11, .14, .37, .29, .09]),
    ]:
        frame = paper[key]
        for _, row in frame.iterrows():
            sample = str(row["样品"])
            for old, new in SAMPLE_NAMES.items():
                sample = sample.replace(old, new)
            score = float(np.dot(row[columns].to_numpy(dtype=float), weights))
            records.append({
                "experiment": experiment, "sample": sample, "score": score,
                "stored_score": float(row["综合评分"]),
                "weighting": "fixed reported weights",
            })
    return pd.DataFrame(records)


def aroma_indices() -> pd.DataFrame:
    """Recompute estimated OAV and within-sample ROAV; preserve non-detections."""
    paper = load_bundle("paper")
    names = json.loads(resource_path("compound_names.json").read_text(encoding="utf-8"))
    records = []
    for experiment, key, samples in [
        ("cultivar", "vol_var", ["HN", "BT", "HT"]),
        ("thermal", "vol_heat", ["WB", "HK", "ZZ", "SZ"]),
    ]:
        frame = paper[key]
        threshold = frame["统一阈值"] if "统一阈值" in frame else frame["阈值"]
        threshold = pd.to_numeric(threshold, errors="coerce").where(lambda s: s > 0)
        for sample in samples:
            oav = frame[f"{sample}_均值"] / threshold
            max_oav = oav.max()
            roav = 100 * oav / max_oav if max_oav > 0 else oav * np.nan
            for index, row in frame.iterrows():
                records.append({
                    "experiment": experiment, "sample": SAMPLE_NAMES.get(sample, sample),
                    "compound_code": row["编号"],
                    "compound": names.get(row["化合物"], row["化合物"]),
                    "threshold_ug_L": threshold.loc[index],
                    "estimated_oav": oav.loc[index], "roav": roav.loc[index],
                })
    return pd.DataFrame(records)


def analyze(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "variance_decomposition.csv": variance_table(),
        "volatile_class_summary.csv": class_summary(),
        "composite_scores.csv": composite_table(),
        "estimated_aroma_indices.csv": aroma_indices(),
    }
    written = []
    for name, frame in tables.items():
        destination = output_dir / name
        frame.to_csv(destination, index=False, encoding="utf-8")
        written.append(destination)
    summary = output_dir / "analysis_scope.json"
    summary.write_text(json.dumps({
        "data_level": "processed treatment means and archived analysis arrays",
        "recomputed": ["variance ratios", "chemical-class sums", "fixed-weight scores", "estimated OAV/ROAV"],
        "not_retrained": ["OPLS-DA", "PLS-DA", "XGBoost", "nested cross-validation", "SHAP"],
        "notes": [
            "Class SDs assume zero covariance; no aggregate significance tests are performed.",
            "Thermal sensory Z-scores are retained from the archived analysis.",
            "The binary heating groups are operational, not an isolated causal heat-transfer factor.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    return written + [summary]
