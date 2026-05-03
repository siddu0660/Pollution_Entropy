#!/usr/bin/env python3
"""
Compare Cosine Similarity vs Mutual Information detector outputs (sparse mode).

Focus:
  - same baseline percentage (default: 30.0)
  - malicious percentages available in both result sets
  - MI compared across z-threshold settings

Outputs:
  - line plots and heatmaps under figures/CosineVsMI/baseline_<pct>/
  - CSV summaries used by the plots
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")


COSINE_RE = re.compile(r"epochs_(?P<baseline>\d+(?:\.\d+)?)_(?P<mal>\d+(?:\.\d+)?)_ml_sparse\.json$")
MI_RE = re.compile(
    r"mi_(?P<baseline>\d+(?:\.\d+)?)_(?P<mal>\d+(?:\.\d+)?)_(?P<z>\d+(?:\.\d+)?)_sparse\.json$"
)
MAL_LABEL = "Malicious percentage"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif (not t) and p:
            fp += 1
        elif (not t) and (not p):
            tn += 1
        else:
            fn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def parse_cosine_file(path: str, baseline_filter: float) -> Tuple[float, Dict[str, float]] | None:
    m = COSINE_RE.search(os.path.basename(path))
    if not m:
        return None
    baseline = float(m.group("baseline"))
    if abs(baseline - baseline_filter) > 1e-9:
        return None
    malicious_pct = float(m.group("mal"))

    with open(path, "r") as f:
        data = json.load(f)
    epochs = data.get("epochs", [])
    y_true = [bool(e.get("attacked_ground_truth", False)) for e in epochs]
    y_pred = [bool(e.get("ml_prediction_anomaly", False)) for e in epochs]
    metrics = confusion_metrics(y_true, y_pred)
    return malicious_pct, metrics


def parse_mi_file(path: str, baseline_filter: float) -> Tuple[float, float, Dict[str, float]] | None:
    m = MI_RE.search(os.path.basename(path))
    if not m:
        return None
    baseline = float(m.group("baseline"))
    if abs(baseline - baseline_filter) > 1e-9:
        return None
    malicious_pct = float(m.group("mal"))
    z_thr = float(m.group("z"))

    with open(path, "r") as f:
        data = json.load(f)

    # Prefer explicit summary if present, otherwise recompute from epochs.
    ds = data.get("detection_summary", {})
    if {"true_positives", "false_positives", "true_negatives", "false_negatives"}.issubset(ds.keys()):
        tp = int(ds.get("true_positives", 0))
        fp = int(ds.get("false_positives", 0))
        tn = int(ds.get("true_negatives", 0))
        fn = int(ds.get("false_negatives", 0))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        acc = safe_div(tp + tn, tp + tn + fp + fn)
        metrics = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
        }
    else:
        epochs = data.get("epochs", [])
        y_true = [bool(e.get("attacked_ground_truth", False)) for e in epochs]
        y_pred = [bool(e.get("anomaly_detected", False)) for e in epochs]
        metrics = confusion_metrics(y_true, y_pred)

    return malicious_pct, z_thr, metrics


def write_csv(path: Path, rows: List[Dict[str, float]], headers: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mi-dir",
        default="Mutual_Info/Sparse",
        help="Directory containing MI sparse JSON files",
    )
    parser.add_argument(
        "--cosine-dir",
        default="CAIDA_Dataset/txtD280/AttackSparseCosine/json",
        help="Directory containing cosine sparse JSON files",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=30.0,
        help="Baseline percentage to compare (default: 30.0)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for comparison figures/csv",
    )
    parser.add_argument(
        "--mi-threshold-focus",
        type=float,
        default=2.5,
        help="MI z-threshold to use for the precision/recall side-by-side plot",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    mi_dir = (repo / args.mi_dir).resolve()
    cosine_dir = (repo / args.cosine_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (repo / "figures" / "CosineVsMI" / f"baseline_{args.baseline:.1f}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cosine_rows: List[Dict[str, float]] = []
    cosine_metrics_by_mal: Dict[float, Dict[str, float]] = {}
    for fn in sorted(glob.glob(str(cosine_dir / "*.json"))):
        parsed = parse_cosine_file(fn, args.baseline)
        if parsed is None:
            continue
        mal, met = parsed
        cosine_metrics_by_mal[mal] = met
        cosine_rows.append(
            {
                "method": "cosine",
                "baseline_pct": args.baseline,
                "malicious_pct": mal,
                "z_threshold": np.nan,
                **met,
            }
        )

    mi_rows: List[Dict[str, float]] = []
    mi_metrics_by_pair: Dict[Tuple[float, float], Dict[str, float]] = {}
    mi_thresholds: set[float] = set()
    for fn in sorted(glob.glob(str(mi_dir / "*.json"))):
        parsed = parse_mi_file(fn, args.baseline)
        if parsed is None:
            continue
        mal, z_thr, met = parsed
        mi_metrics_by_pair[(mal, z_thr)] = met
        mi_thresholds.add(z_thr)
        mi_rows.append(
            {
                "method": "mi",
                "baseline_pct": args.baseline,
                "malicious_pct": mal,
                "z_threshold": z_thr,
                **met,
            }
        )

    if not cosine_metrics_by_mal:
        raise SystemExit(f"No cosine files found for baseline={args.baseline} in {cosine_dir}")
    if not mi_metrics_by_pair:
        raise SystemExit(f"No MI files found for baseline={args.baseline} in {mi_dir}")

    write_csv(
        out_dir / "cosine_metrics.csv",
        cosine_rows,
        ["method", "baseline_pct", "malicious_pct", "z_threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"],
    )
    write_csv(
        out_dir / "mi_metrics.csv",
        mi_rows,
        ["method", "baseline_pct", "malicious_pct", "z_threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"],
    )

    common_mal = sorted(set(cosine_metrics_by_mal.keys()) & {k[0] for k in mi_metrics_by_pair.keys()})
    thresholds = sorted(mi_thresholds)
    if not common_mal:
        raise SystemExit("No overlapping malicious percentages between cosine and MI files.")

    # Plot 1: F1 vs malicious percentage
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        common_mal,
        [cosine_metrics_by_mal[m]["f1"] for m in common_mal],
        marker="o",
        lw=2.2,
        color="#1f77b4",
        label="Cosine",
    )
    for z in thresholds:
        y = [mi_metrics_by_pair[(m, z)]["f1"] for m in common_mal if (m, z) in mi_metrics_by_pair]
        x = [m for m in common_mal if (m, z) in mi_metrics_by_pair]
        if x:
            ax.plot(x, y, marker="s", lw=1.8, label=f"MI (z={z:g})")
    ax.set_title(f"F1 comparison vs malicious percentage (baseline {args.baseline:.1f}%)")
    ax.set_xlabel(MAL_LABEL)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "f1_vs_malicious.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: MI F1 heatmap (rows=malicious, cols=z-threshold)
    mi_f1 = np.full((len(common_mal), len(thresholds)), np.nan, dtype=float)
    for i, m in enumerate(common_mal):
        for j, z in enumerate(thresholds):
            rec = mi_metrics_by_pair.get((m, z))
            if rec is not None:
                mi_f1[i, j] = rec["f1"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.heatmap(
        mi_f1,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[f"{z:g}" for z in thresholds],
        yticklabels=[f"{m:g}" for m in common_mal],
        cbar_kws={"label": "MI F1"},
        ax=ax,
    )
    ax.set_title(f"MI F1 heatmap (baseline {args.baseline:.1f}%)")
    ax.set_xlabel("MI z-threshold")
    ax.set_ylabel(MAL_LABEL)
    fig.tight_layout()
    fig.savefig(out_dir / "mi_f1_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: MI-Cosine F1 gain heatmap by (malicious, z-threshold)
    f1_gain = np.full((len(common_mal), len(thresholds)), np.nan, dtype=float)
    for i, m in enumerate(common_mal):
        c_f1 = cosine_metrics_by_mal[m]["f1"]
        for j, z in enumerate(thresholds):
            rec = mi_metrics_by_pair.get((m, z))
            if rec is not None:
                f1_gain[i, j] = rec["f1"] - c_f1
    vmax = float(np.nanmax(np.abs(f1_gain))) if np.isfinite(f1_gain).any() else 1.0
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.heatmap(
        f1_gain,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[f"{z:g}" for z in thresholds],
        yticklabels=[f"{m:g}" for m in common_mal],
        cbar_kws={"label": "MI F1 - Cosine F1"},
        ax=ax,
    )
    ax.set_title(f"F1 gain heatmap vs Cosine (baseline {args.baseline:.1f}%)")
    ax.set_xlabel("MI z-threshold")
    ax.set_ylabel(MAL_LABEL)
    fig.tight_layout()
    fig.savefig(out_dir / "mi_minus_cosine_f1_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Precision/Recall side-by-side for a chosen MI threshold (fallback to first available)
    chosen_z = args.mi_threshold_focus if args.mi_threshold_focus in thresholds else thresholds[0]
    pr_fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    axes[0].plot(
        common_mal,
        [cosine_metrics_by_mal[m]["precision"] for m in common_mal],
        marker="o",
        lw=2.0,
        label="Cosine",
    )
    axes[0].plot(
        common_mal,
        [mi_metrics_by_pair[(m, chosen_z)]["precision"] for m in common_mal],
        marker="s",
        lw=2.0,
        label=f"MI (z={chosen_z:g})",
    )
    axes[0].set_title("Precision")
    axes[0].set_xlabel(MAL_LABEL)
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        common_mal,
        [cosine_metrics_by_mal[m]["recall"] for m in common_mal],
        marker="o",
        lw=2.0,
        label="Cosine",
    )
    axes[1].plot(
        common_mal,
        [mi_metrics_by_pair[(m, chosen_z)]["recall"] for m in common_mal],
        marker="s",
        lw=2.0,
        label=f"MI (z={chosen_z:g})",
    )
    axes[1].set_title("Recall")
    axes[1].set_xlabel(MAL_LABEL)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.set_ylim(0.0, 1.02)
    pr_fig.suptitle(f"Cosine vs MI @ baseline {args.baseline:.1f}%")
    pr_fig.tight_layout()
    pr_fig.savefig(out_dir / f"precision_recall_vs_malicious_mi_z_{chosen_z:g}.png", dpi=220, bbox_inches="tight")
    plt.close(pr_fig)

    print(f"Wrote comparison artifacts to: {out_dir}")
    print(f"Overlapping malicious percentages: {common_mal}")
    print(f"MI thresholds found: {thresholds}")


if __name__ == "__main__":
    main()
