"""
Plot_MI_Results.py — Visualization for Mutual Information Detector
===================================================================

Generates comprehensive plots for Entropy_MI_Detector.py JSON outputs:
  1. MI feature pair timelines (observed vs baseline)
  2. Z-score timelines with detection threshold bands
  3. Detection flags (attacked vs detected)
  4. Confusion matrix (sparse mode)
  5. Feature importance (which pairs trigger most detections)
  6. ROC curve (if multiple threshold runs available)
  7. Multi-file comparison

Usage:
------
  # Single file
  python Plot_MI_Results.py results.json

  # Multiple files (comparison)
  python Plot_MI_Results.py result1.json result2.json result3.json

  # Entire directory
  python Plot_MI_Results.py /path/to/json_dir/

  # Save to specific file
  python Plot_MI_Results.py results.json --output analysis.pdf

  # Show plots interactively
  python Plot_MI_Results.py results.json --show

  # 10-epoch attack/detection windows (default 10; change with --window-size N)
  python Plot_MI_Results.py results.json --window-size 10 -o report.pdf
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found — only text summary available.")
    print("Install: pip install matplotlib numpy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MI_FEATURES = [
    "mi_src_dst_ip",
    "mi_src_ip_dst_port",
    "mi_src_ip_src_port",
    "mi_dst_ip_dst_port",
    "mi_src_dst_port",
    "mi_protocol_dst_port"
]

ENTROPY_FEATURES = ["h_src_ip", "h_dst_ip", "h_src_port", "h_dst_port"]

NICE_NAMES = {
    "mi_src_dst_ip": "MI(Src-IP, Dst-IP)",
    "mi_src_ip_dst_port": "MI(Src-IP, Dst-Port)",
    "mi_src_ip_src_port": "MI(Src-IP, Src-Port)",
    "mi_dst_ip_dst_port": "MI(Dst-IP, Dst-Port)",
    "mi_src_dst_port": "MI(Src-Port, Dst-Port)",
    "mi_protocol_dst_port": "MI(Protocol, Dst-Port)",
    "h_src_ip": "H(Src-IP)",
    "h_dst_ip": "H(Dst-IP)",
    "h_src_port": "H(Src-Port)",
    "h_dst_port": "H(Dst-Port)",
}

# Colors
CLR_BASELINE = "#6A9BD8"
CLR_OBSERVED = "#E85D75"
CLR_DETECTED = "#FF6B6B"
CLR_MISSED = "#95E1D3"
CLR_CLEAN = "#A8E6CF"
CLR_ATTACKED_BG = "#FFE5E5"
CLR_THRESHOLD = "#FFB84D"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _looks_like_mi_output(data: dict) -> bool:
    """Heuristic: check if JSON resembles Entropy_MI_Detector output."""
    if not isinstance(data, dict):
        return False
    if "metadata" not in data or "epochs" not in data or "baseline" not in data:
        return False
    epochs = data.get("epochs", [])
    if not isinstance(epochs, list) or not epochs:
        return False
    first = epochs[0]
    if not isinstance(first, dict):
        return False
    return isinstance(first.get("mi_metrics"), dict) and isinstance(first.get("z_scores"), dict)


def discover_json_files(inputs: List[str]) -> List[str]:
    """
    Expand input paths into a list of JSON file paths.
    - If a path is a file, keep it.
    - If a path is a directory, recursively search for *.json (prioritise *mi*.json).
    """
    out: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            root = Path(p)
            mi_first = sorted(root.rglob("*mi*.json"))
            all_json = sorted(root.rglob("*.json"))
            seen = set()
            for fn in mi_first + all_json:
                s = str(fn)
                if s not in seen:
                    out.append(s)
                    seen.add(s)
        else:
            out.append(p)
    return out


def collect_jsons(paths: List[str], verbose: bool = False) -> List[Tuple[str, dict]]:
    """Load MI-output JSONs from files/directories (recursive)."""
    result: List[Tuple[str, dict]] = []
    rejected: List[str] = []
    for fn in discover_json_files(paths):
        if not os.path.isfile(fn):
            continue
        try:
            data = load_json(fn)
        except Exception as ex:
            if verbose:
                print(f"  [skip] {fn}: {ex}")
            continue
        if _looks_like_mi_output(data):
            result.append((fn, data))
        else:
            rejected.append(fn)
    if not result and rejected:
        print("  JSON files found but none matched Entropy_MI_Detector schema "
              "(need metadata, baseline, epochs[].mi_metrics, epochs[].z_scores).")
        show = rejected if verbose else rejected[:5]
        for fn in show:
            print(f"    rejected: {fn}")
        if not verbose and len(rejected) > 5:
            print(f"    ... and {len(rejected) - 5} more (use --verbose for full list)")
    return result


def is_sparse(data: dict) -> bool:
    return data.get("metadata", {}).get("sparse_mode", False)


def get_title(data: dict) -> str:
    m = data.get("metadata", {})
    title = f"{m.get('dataset_name', '?')}"
    title += f" | Baseline {m.get('baseline_percentage', '?')}%"
    title += f", Malicious {m.get('malicious_percentage', '?')}%"
    if m.get("sparse_mode"):
        title += " [SPARSE]"
    return title


def epoch_ground_truth(e: dict) -> Optional[bool]:
    """
    Some outputs use 'attacked_ground_truth' (sparse mode),
    others use 'attacked' (every-epoch injection runs).
    """
    if "attacked_ground_truth" in e:
        return bool(e.get("attacked_ground_truth"))
    if "attacked" in e:
        return bool(e.get("attacked"))
    return None


def epochs_have_attack_labels(epochs: List[dict]) -> bool:
    return any(epoch_ground_truth(e) is not None for e in epochs)


def sort_epochs_chronologically(epochs: List[dict]) -> List[dict]:
    """Stable sort by epoch_index (falls back to list order)."""
    def key_fn(e: dict):
        v = e.get("epoch_index")
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, 0)

    return sorted(epochs, key=key_fn)


def pick_top_mi_pair(data: dict, epoch: dict) -> Tuple[str, float, float]:
    """
    Pick the most informative MI pair for reporting for this epoch:
    highest |z| among MI pairs whose baseline std > 0.
    Returns: (feature_key, mi_value, z_value)
    """
    baseline = data.get("baseline", {})
    z = epoch.get("z_scores", {}) or {}
    mi = epoch.get("mi_metrics", {}) or {}

    best_feat = MI_FEATURES[0]
    best_abs_z = -1.0
    best_mi_val = float(mi.get(best_feat, float("nan")))
    best_z_val = float(z.get(f"{best_feat}_z", 0.0))

    for feat in MI_FEATURES:
        std = float(baseline.get(f"{feat}_std", 0.0) or 0.0)
        if std <= 0.0:
            # Avoid selecting unstable z-scores (division by ~0) like protocol/dst_port
            continue
        z_val = float(z.get(f"{feat}_z", 0.0))
        if abs(z_val) > best_abs_z:
            best_abs_z = abs(z_val)
            best_feat = feat
            best_z_val = z_val
            best_mi_val = float(mi.get(feat, float("nan")))

    return best_feat, best_mi_val, best_z_val


# ---------------------------------------------------------------------------
# Text Summary
# ---------------------------------------------------------------------------
def print_summary(data: dict, label: str):
    meta = data.get("metadata", {})
    epochs = data.get("epochs", [])
    summary = data.get("detection_summary", {})

    print("\n" + "=" * 80)
    print(f"  FILE: {label}")
    print("=" * 80)
    print(f"  Dataset:     {meta.get('dataset_name', '?')}")
    print(f"  Method:      {meta.get('method', 'MI Detector')}")
    print(f"  Baseline:    {meta.get('baseline_percentage', '?')}%")
    print(f"  Malicious:   {meta.get('malicious_percentage', '?')}%")
    print(f"  Sparse mode: {meta.get('sparse_mode', False)}")
    print(f"  Z-threshold: {meta.get('z_threshold', '?')}")
    print(f"  Total epochs analyzed: {len(epochs)}")

    # Detection summary
    if is_sparse(data):
        print("\n  --- Confusion Matrix ---")
        tp = summary.get("true_positives", 0)
        fp = summary.get("false_positives", 0)
        tn = summary.get("true_negatives", 0)
        fn = summary.get("false_negatives", 0)

        print(f"  True Positives:  {tp:4d}")
        print(f"  False Positives: {fp:4d}")
        print(f"  True Negatives:  {tn:4d}")
        print(f"  False Negatives: {fn:4d}")
        print(f"\n  Precision: {summary.get('precision', 0):.2%}")
        print(f"  Recall:    {summary.get('recall', 0):.2%}")
        print(f"  Accuracy:  {summary.get('accuracy', 0):.2%}")
    else:
        detected = summary.get("detected_count", 0)
        total = summary.get("total_epochs", len(epochs))
        rate = summary.get("detection_rate", 0)
        print(f"\n  Detected: {detected}/{total} ({rate:.1%})")

    # Most common detection reasons
    if epochs:
        reasons = defaultdict(int)
        for e in epochs:
            if e.get("anomaly_detected"):
                reason = e.get("detection_reason", "unknown")
                reasons[reason] += 1

        if reasons:
            print("\n  --- Detection Triggers (top 5) ---")
            for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {reason:40s}: {count:3d}")

    # MI statistics
    baseline = data.get("baseline", {})
    if baseline:
        print("\n  --- Baseline MI Statistics ---")
        print(f"  {'Feature Pair':<30s} {'Mean':>10s} {'Std':>10s}")
        print("  " + "-" * 52)
        for feat in MI_FEATURES:
            mean = baseline.get(f"{feat}_mean", 0)
            std = baseline.get(f"{feat}_std", 0)
            print(f"  {NICE_NAMES.get(feat, feat):<30s} {mean:10.4f} {std:10.4f}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
if HAS_MPL:
    # -----------------------------------------------------------------------
    # "Epoch-style" plotting helpers (match Helpers/generatePlotsEpoch.py look)
    # -----------------------------------------------------------------------
    EPOCHSTYLE_LINE = "#2E4057"
    EPOCHSTYLE_POS = "#E63946"   # positive change
    EPOCHSTYLE_NEG = "#06A77D"   # negative change

    def group_epochs(epoch_indices: List[int], values: List[float], group_size: int = 10):
        """Group epochs into fixed-size groups and average the values."""
        if not epoch_indices:
            return [], []
        num_groups = (len(epoch_indices) + group_size - 1) // group_size
        grouped_labels: List[str] = []
        grouped_values: List[float] = []
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, len(epoch_indices))
            grouped_labels.append(f"{epoch_indices[start]}-{epoch_indices[end-1]}")
            grouped_values.append(float(np.mean(values[start:end])) if end > start else 0.0)
        return grouped_labels, grouped_values

    def group_epochs_with_attack(
        epoch_indices: List[int],
        values: List[float],
        attacked_flags: List[bool],
        group_size: int = 10,
    ) -> Tuple[List[str], List[float], List[bool]]:
        """
        Same as group_epochs, plus per-window flag: True if any epoch in the window is under attack (GT).
        """
        if not epoch_indices or len(values) != len(epoch_indices) or len(attacked_flags) != len(epoch_indices):
            return [], [], []
        num_groups = (len(epoch_indices) + group_size - 1) // group_size
        grouped_labels: List[str] = []
        grouped_values: List[float] = []
        window_attacked: List[bool] = []
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, len(epoch_indices))
            grouped_labels.append(f"{epoch_indices[start]}-{epoch_indices[end-1]}")
            grouped_values.append(float(np.mean(values[start:end])) if end > start else 0.0)
            window_attacked.append(any(attacked_flags[start:end]))
        return grouped_labels, grouped_values, window_attacked


    def create_percentage_plot(
        ax,
        x_labels: List[str],
        percentage_values: List[float],
        title: str,
        window_attacked: Optional[List[bool]] = None,
    ):
        """
        Clean line plot: percentage difference from baseline, using the same
        visual style as Helpers/generatePlotsEpoch.py.
        """
        x_pos = np.arange(len(x_labels))

        if window_attacked and len(window_attacked) == len(x_labels):
            for i, has_attack in enumerate(window_attacked):
                if has_attack:
                    ax.axvspan(
                        i - 0.5,
                        i + 0.5,
                        facecolor="#FFADAD",
                        alpha=0.45,
                        zorder=0,
                        linewidth=0,
                    )

        ax.plot(
            x_pos,
            percentage_values,
            color=EPOCHSTYLE_LINE,
            linewidth=2.5,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=EPOCHSTYLE_LINE,
            alpha=0.9,
        )

        for x, y in zip(x_pos, percentage_values):
            color = EPOCHSTYLE_POS if y > 0 else EPOCHSTYLE_NEG
            ax.plot(x, y, "o", markersize=8, color=color, alpha=0.85, zorder=3)

        ax.axhline(y=0, color="#333333", linestyle="-", linewidth=1.5, alpha=0.6, zorder=1)

        ax.set_xlabel("Epoch Groups", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage Change (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title} — % Difference from Baseline", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.15, linestyle="--", linewidth=0.5)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=EPOCHSTYLE_POS,
                   markersize=8, label="Increase from Baseline", alpha=0.85),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=EPOCHSTYLE_NEG,
                   markersize=8, label="Decrease from Baseline", alpha=0.85),
        ]
        if window_attacked and any(window_attacked):
            legend_elements.append(
                mpatches.Patch(facecolor="#FFADAD", alpha=0.55, edgecolor="none",
                               label="Window has ≥1 attacked epoch (GT)")
            )
        ax.legend(handles=legend_elements, loc="best", fontsize=10, framealpha=0.95)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)


    def plot_epochstyle_mi_pages(pdf: PdfPages, data: dict, group_size: int = 10):
        """
        Produce epoch-style pages (one per MI pair) where the y-axis is
        percentage change from the baseline mean:
            pct = ((observed - mean) / mean) * 100
        """
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        baseline = data.get("baseline", {}) or {}
        if not epochs:
            return

        indices: List[int] = []
        attacked_flags: List[bool] = []
        for i, e in enumerate(epochs):
            ei = e.get("epoch_index", None)
            try:
                indices.append(int(ei) if ei is not None else i)
            except (TypeError, ValueError):
                indices.append(i)
            gt = epoch_ground_truth(e)
            attacked_flags.append(bool(gt) if gt is not None else False)

        for feat in MI_FEATURES:
            base_mean = float(baseline.get(f"{feat}_mean", 0.0) or 0.0)
            observed = []
            for e in epochs:
                v = (e.get("mi_metrics", {}) or {}).get(feat, float("nan"))
                observed.append(float(v) if v is not None else float("nan"))

            pct = []
            for v in observed:
                if base_mean != 0 and not math.isnan(v):
                    pct.append(((v - base_mean) / base_mean) * 100.0)
                elif math.isnan(v):
                    pct.append(0.0)
                else:
                    pct.append(0.0)

            x_labels, y_vals, win_attack = group_epochs_with_attack(
                indices, pct, attacked_flags, group_size=group_size
            )
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            create_percentage_plot(
                ax, x_labels, y_vals, NICE_NAMES.get(feat, feat), window_attacked=win_attack
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def _render_text_page(pdf: PdfPages, title: str, lines: List[str]):
        fig = plt.figure(figsize=(11.0, 8.5))  # landscape letter-ish
        fig.patch.set_facecolor("white")
        plt.axis("off")
        plt.text(0.02, 0.96, title, fontsize=16, fontweight="bold", va="top")
        y = 0.90
        for ln in lines:
            plt.text(0.02, y, ln, fontsize=11, va="top")
            y -= 0.035
            if y < 0.05:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    def _render_table_page(
        pdf: PdfPages,
        title: str,
        col_labels: List[str],
        rows: List[List[str]],
        note: Optional[str] = None,
    ):
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        plt.text(0.02, 0.96, title, fontsize=14, fontweight="bold", va="top")
        if note:
            plt.text(0.02, 0.92, note, fontsize=10, color="#444444", va="top")

        table = plt.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            colLoc="center",
            loc="upper left",
            bbox=[0.02, 0.05, 0.96, 0.84],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor("#f0f2f5")
                cell.set_text_props(fontweight="bold", color="#222222")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    def plot_attack_recognition_windows(pdf: PdfPages, data: dict, group_size: int = 10):
        """
        One row per chronological window of `group_size` epochs.
        Cell fill: red = ground-truth attack, gray = clean.
        Text: TP / FN / FP / TN (detector vs ground truth).
        Border highlights detector fire (thick when anomaly_detected).
        """
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        if not epochs:
            return
        if not epochs_have_attack_labels(epochs):
            _render_text_page(
                pdf,
                "Attack / detection by epoch windows",
                [
                    "No per-epoch attack labels in this JSON.",
                    "Expected: attacked (every-epoch runs) or attacked_ground_truth (sparse).",
                ],
            )
            return

        n = len(epochs)
        n_win = (n + group_size - 1) // group_size
        fig_h = min(24.0, max(5.5, 0.38 * n_win + 2.8))
        fig, ax = plt.subplots(figsize=(15.5, fig_h))

        ax.set_xlim(-1.35, group_size + 0.45)
        ax.set_ylim(-0.55, n_win - 0.45)
        ax.invert_yaxis()

        from matplotlib.patches import Rectangle

        for wi in range(n_win):
            start = wi * group_size
            end = min(start + group_size, n)
            chunk = epochs[start:end]
            try:
                ep_a = int(chunk[0].get("epoch_index", start))
                ep_b = int(chunk[-1].get("epoch_index", end - 1))
            except (TypeError, ValueError):
                ep_a, ep_b = start, end - 1

            ax.text(
                -1.28,
                wi,
                f"W{wi + 1}\n{ep_a}–{ep_b}",
                ha="right",
                va="center",
                fontsize=9,
                linespacing=1.05,
                color="#333333",
            )

            tp = fp = tn = fn = 0
            for j, e in enumerate(chunk):
                gt = epoch_ground_truth(e)
                det = bool(e.get("anomaly_detected"))
                if gt is None:
                    outcome = "?"
                    face = "#DEE2E6"
                    edge, lw = "#6C757D", 1.2
                elif gt is True and det:
                    outcome, tp = "TP", tp + 1
                    face = "#C1121F"
                    edge, lw = "#2D6A4F", 2.8
                elif gt is True and not det:
                    outcome, fn = "FN", fn + 1
                    face = "#C1121F"
                    edge, lw = "#E85D04", 2.8
                elif gt is False and det:
                    outcome, fp = "FP", fp + 1
                    face = "#E9ECEF"
                    edge, lw = "#F4A261", 2.6
                else:
                    outcome, tn = "TN", tn + 1
                    face = "#E9ECEF"
                    edge, lw = "#ADB5BD", 1.0

                rect = Rectangle(
                    (j - 0.48, wi - 0.48),
                    0.96,
                    0.96,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=lw,
                    zorder=2,
                )
                ax.add_patch(rect)
                attacked = gt is True
                txt_color = "#F8F9FA" if attacked else "#212529"
                ax.text(j, wi, outcome, ha="center", va="center", fontsize=10,
                        fontweight="bold", color=txt_color, zorder=3)

            summary = f"TP {tp}  FN {fn}  FP {fp}  TN {tn}"
            ax.text(
                group_size + 0.08,
                wi,
                summary,
                ha="left",
                va="center",
                fontsize=8.5,
                color="#495057",
                family="monospace",
            )

        ax.set_xticks(np.arange(group_size) + 0.0)
        ax.set_xticklabels([str(i + 1) for i in range(group_size)], fontsize=10)
        ax.set_xlabel(
            f"Epoch slot within each {group_size}-epoch window (chronological order)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        title = (
            f"Ground-truth attack (red) vs detector — {group_size}-epoch windows — "
            f"{get_title(data)}"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

        legend_elems = [
            mpatches.Patch(facecolor="#C1121F", edgecolor="#2D6A4F", linewidth=2,
                           label="TP: under attack, flagged"),
            mpatches.Patch(facecolor="#C1121F", edgecolor="#E85D04", linewidth=2,
                           label="FN: under attack, missed"),
            mpatches.Patch(facecolor="#E9ECEF", edgecolor="#F4A261", linewidth=2,
                           label="FP: clean, flagged"),
            mpatches.Patch(facecolor="#E9ECEF", edgecolor="#ADB5BD", linewidth=1,
                           label="TN: clean, not flagged"),
            mpatches.Patch(facecolor="#DEE2E6", edgecolor="#6C757D", linewidth=1.2,
                           label="?: no attack label on this epoch"),
        ]
        ax.legend(
            handles=legend_elems,
            loc="lower center",
            bbox_to_anchor=(0.45, 1.02),
            ncol=2,
            fontsize=9,
            frameon=True,
            framealpha=0.95,
        )

        note = (
            "Each cell is one epoch. Red = attack present in data (GT). "
            "Letters = overlap with detector. Right column = counts in that window."
        )
        fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#555555")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    def generate_minimal_report(pdf: PdfPages, data: dict, label: str, group_size: int = 10):
        """
        Minimal report (necessary info only):
        - Metadata + baseline threshold
        - Detection performance summary (if ground truth exists)
        - Table of key epochs with: top MI pair value, z-score, detected, actual attack
        """
        meta = data.get("metadata", {})
        epochs = data.get("epochs", []) or []
        baseline = data.get("baseline", {}) or {}
        z_thr = meta.get("z_threshold", 2.5)

        # Summary counts
        gt_list = [epoch_ground_truth(e) for e in epochs]
        has_gt = any(g is not None for g in gt_list)
        det_list = [bool(e.get("anomaly_detected")) for e in epochs]

        tp = fp = tn = fn = 0
        if has_gt:
            for e in epochs:
                gt = epoch_ground_truth(e)
                if gt is None:
                    continue
                det = bool(e.get("anomaly_detected"))
                if gt and det:
                    tp += 1
                elif (not gt) and det:
                    fp += 1
                elif (not gt) and (not det):
                    tn += 1
                elif gt and (not det):
                    fn += 1

        # Page 1: metadata + baseline stats (just MI means/stds)
        lines = [
            f"File: {label}",
            f"Dataset: {meta.get('dataset_name', '?')}",
            f"Method: {meta.get('method', 'Mutual Information Detector')}",
            f"Sparse mode: {bool(meta.get('sparse_mode', False))}",
            f"Baseline %: {meta.get('baseline_percentage', '?')}   Malicious %: {meta.get('malicious_percentage', '?')}",
            f"Z-threshold: {z_thr}",
            "",
            "Baseline MI (mean ± std):",
        ]
        for feat in MI_FEATURES:
            mean = baseline.get(f"{feat}_mean", None)
            std = baseline.get(f"{feat}_std", None)
            if mean is None or std is None:
                continue
            lines.append(f"  {NICE_NAMES.get(feat, feat)}: {float(mean):.4f} ± {float(std):.4f}")

        if has_gt:
            lines += [
                "",
                f"Performance (where ground-truth exists): TP={tp}, FP={fp}, TN={tn}, FN={fn}",
            ]
        else:
            detected_cnt = sum(1 for d in det_list if d)
            lines += ["", f"Detected anomalies: {detected_cnt}/{len(epochs)}"]

        _render_text_page(pdf, "MI Detector — Minimal Analysis", lines)

        # Attack vs detection in fixed-size windows (primary interpretability plot)
        plot_attack_recognition_windows(pdf, data, group_size=group_size)

        # Epoch-style MI plots (same look as Helpers/generatePlotsEpoch.py)
        plot_epochstyle_mi_pages(pdf, data, group_size=group_size)

        # Build table rows (only necessary epochs)
        key_epochs: List[dict] = []
        for e in epochs:
            gt = epoch_ground_truth(e)
            det = bool(e.get("anomaly_detected"))
            if has_gt:
                if gt or det:
                    key_epochs.append(e)
            else:
                if det:
                    key_epochs.append(e)

        # If nothing detected and no GT, still show first few epochs as reference
        if not key_epochs and epochs:
            key_epochs = epochs[: min(15, len(epochs))]

        # Paginate (keep PDFs readable)
        page_size = 35
        for page_start in range(0, len(key_epochs), page_size):
            chunk = key_epochs[page_start : page_start + page_size]
            rows: List[List[str]] = []
            for e in chunk:
                idx = e.get("epoch_index", "?")
                gt = epoch_ground_truth(e)
                det = bool(e.get("anomaly_detected"))
                top_feat, top_mi, top_z = pick_top_mi_pair(data, e)

                rows.append([
                    str(idx),
                    "Yes" if gt is True else ("No" if gt is False else "—"),
                    "Yes" if det else "No",
                    NICE_NAMES.get(top_feat, top_feat),
                    f"{top_mi:.4f}" if isinstance(top_mi, (int, float)) and not math.isnan(top_mi) else "—",
                    f"{top_z:.2f}",
                    f"{float(e.get('max_z_score', 0.0)):.2f}",
                ])

            note = "Rows shown: attacked or detected (sparse), otherwise detected-only. Top MI pair = highest |z| (excluding pairs with baseline std=0)."
            _render_table_page(
                pdf,
                f"Key Epochs ({page_start+1}-{min(page_start+page_size, len(key_epochs))} of {len(key_epochs)})",
                ["Epoch", "Actual attack", "Detected", "Top MI pair", "MI value", "Z (pair)", "Max |Z|"],
                rows,
                note=note,
            )

    def plot_mi_timelines(pdf: PdfPages, data: dict):
        """Plot MI values over epochs with baseline bands"""
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        if not epochs:
            return

        baseline = data.get("baseline", {})
        indices = []
        for e in epochs:
            v = e.get("epoch_index")
            try:
                indices.append(int(v) if v is not None else len(indices))
            except (TypeError, ValueError):
                indices.append(len(indices))

        # Create subplots for each MI feature pair
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for idx, feat in enumerate(MI_FEATURES):
            ax = axes[idx]

            # Get observed values
            values = [e["mi_metrics"].get(feat, float('nan')) for e in epochs]

            # Baseline mean and std
            baseline_mean = baseline.get(f"{feat}_mean", 0)
            baseline_std = baseline.get(f"{feat}_std", 0)

            # Light red band on ground-truth attacked epochs (sparse or every-epoch)
            for e in epochs:
                if epoch_ground_truth(e) is True:
                    xi = e.get("epoch_index")
                    try:
                        x0 = float(xi)
                    except (TypeError, ValueError):
                        continue
                    ax.axvspan(x0 - 0.5, x0 + 0.5, color="#F8B4B4", alpha=0.55, linewidth=0, zorder=0)

            # Baseline band (mean ± 2*std)
            ax.axhline(baseline_mean, color=CLR_BASELINE, linestyle='--',
                      linewidth=1.5, label='Baseline Mean', alpha=0.8)
            ax.fill_between(indices,
                           baseline_mean - 2*baseline_std,
                           baseline_mean + 2*baseline_std,
                           color=CLR_BASELINE, alpha=0.15, label='±2σ band')

            # Observed values
            detected_indices = [indices[i] for i, e in enumerate(epochs) if e.get("anomaly_detected")]
            detected_values = [
                e["mi_metrics"].get(feat, float("nan")) for e in epochs if e.get("anomaly_detected")
            ]

            ax.plot(indices, values, 'o-', color=CLR_OBSERVED,
                   markersize=4, linewidth=1.2, label='Observed', alpha=0.8)

            # Highlight detected anomalies
            if detected_indices:
                ax.scatter(detected_indices, detected_values,
                          color=CLR_DETECTED, s=80, marker='X',
                          edgecolors='black', linewidth=1,
                          label='Detected', zorder=5)

            ax.set_title(NICE_NAMES.get(feat, feat), fontsize=10, fontweight='bold')
            ax.set_ylabel('MI (bits)', fontsize=8)
            ax.grid(alpha=0.3)

            if idx >= 4:
                ax.set_xlabel('Epoch Index', fontsize=8)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4,
                  fontsize=9, bbox_to_anchor=(0.5, -0.02))

        st = f"MI Feature Pairs Timeline — {get_title(data)}"
        if epochs_have_attack_labels(epochs):
            st += "\n(Pink vertical band = ground-truth attack epoch)"
        fig.suptitle(st, fontsize=11, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_z_scores(pdf: PdfPages, data: dict):
        """Plot Z-scores with threshold bands"""
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        if not epochs:
            return

        threshold = data.get("metadata", {}).get("z_threshold", 2.5)
        indices = []
        for e in epochs:
            v = e.get("epoch_index")
            try:
                indices.append(int(v) if v is not None else len(indices))
            except (TypeError, ValueError):
                indices.append(len(indices))

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for idx, feat in enumerate(MI_FEATURES):
            ax = axes[idx]
            z_key = f"{feat}_z"

            # Get Z-scores
            z_scores = [e["z_scores"].get(z_key, 0) for e in epochs]

            for e in epochs:
                if epoch_ground_truth(e) is True:
                    xi = e.get("epoch_index")
                    try:
                        x0 = float(xi)
                    except (TypeError, ValueError):
                        continue
                    ax.axvspan(x0 - 0.5, x0 + 0.5, color="#F8B4B4", alpha=0.55, linewidth=0, zorder=0)

            # Threshold bands
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.axhline(threshold, color=CLR_THRESHOLD, linestyle='--',
                      linewidth=1.5, label=f'Threshold ±{threshold}')
            ax.axhline(-threshold, color=CLR_THRESHOLD, linestyle='--', linewidth=1.5)
            ax.fill_between(indices, threshold, 10, color=CLR_THRESHOLD, alpha=0.1)
            ax.fill_between(indices, -threshold, -10, color=CLR_THRESHOLD, alpha=0.1)

            # Z-scores
            colors = [CLR_DETECTED if abs(z) > threshold else CLR_OBSERVED
                     for z in z_scores]
            ax.scatter(indices, z_scores, c=colors, s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.plot(indices, z_scores, '-', color=CLR_OBSERVED, alpha=0.3, linewidth=1)

            ax.set_title(NICE_NAMES.get(feat, feat), fontsize=10, fontweight='bold')
            ax.set_ylabel('Z-score', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-5, 5)

            if idx >= 4:
                ax.set_xlabel('Epoch Index', fontsize=8)

        st = f"MI Z-Scores — {get_title(data)}"
        if epochs_have_attack_labels(epochs):
            st += "\n(Pink vertical band = ground-truth attack epoch)"
        fig.suptitle(st, fontsize=11, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_detection_overview(pdf: PdfPages, data: dict):
        """Overview: detected vs ground truth (sparse or every-epoch labeled runs)."""
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        if not epochs or not epochs_have_attack_labels(epochs):
            return

        indices = []
        for e in epochs:
            v = e.get("epoch_index")
            try:
                indices.append(int(v) if v is not None else len(indices))
            except (TypeError, ValueError):
                indices.append(len(indices))

        ground_truth = [1 if epoch_ground_truth(e) else 0 for e in epochs]
        detected = [1 if e.get("anomaly_detected") else 0 for e in epochs]
        max_z_scores = [e.get("max_z_score", 0) for e in epochs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 2]})

        # Top: Binary flags
        width = 0.35
        x = np.arange(len(indices))

        bars1 = ax1.bar(x - width/2, ground_truth, width, label='Ground Truth (Attacked)',
                       color='#FF9999', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, detected, width, label='Detected (Anomaly)',
                       color='#6699FF', alpha=0.7, edgecolor='black', linewidth=0.5)

        ax1.set_ylabel('Binary Flag', fontsize=9)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['No', 'Yes'])
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_title('Detection Flags', fontsize=10, fontweight='bold')

        # Bottom: Max Z-scores
        threshold = data.get("metadata", {}).get("z_threshold", 2.5)

        colors = []
        for gt, det in zip(ground_truth, detected):
            if gt and det:
                colors.append('#00AA00')  # True positive - green
            elif gt and not det:
                colors.append('#FF6600')  # False negative - orange
            elif not gt and det:
                colors.append('#FFAA00')  # False positive - yellow
            else:
                colors.append('#AAAAAA')  # True negative - gray

        ax2.bar(x, max_z_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(threshold, color=CLR_THRESHOLD, linestyle='--',
                   linewidth=2, label=f'Threshold = {threshold}')
        ax2.set_ylabel('Max |Z-score|', fontsize=9)
        ax2.set_xlabel('Epoch Index', fontsize=9)
        ax2.set_xticks(x[::max(1, len(x)//20)])
        ax2.set_xticklabels([indices[i] for i in range(0, len(indices), max(1, len(x)//20))],
                           rotation=45)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)

        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#00AA00', label='True Positive'),
            Patch(facecolor='#FFAA00', label='False Positive'),
            Patch(facecolor='#FF6600', label='False Negative'),
            Patch(facecolor='#AAAAAA', label='True Negative')
        ]
        ax2.legend(handles=legend_elements, fontsize=8, loc='upper right', ncol=4)

        fig.suptitle(f"Detection Overview — {get_title(data)}",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_confusion_matrix(pdf: PdfPages, data: dict):
        """Confusion matrix heatmap"""
        if not is_sparse(data):
            return

        summary = data.get("detection_summary", {})
        tp = summary.get("true_positives", 0)
        fp = summary.get("false_positives", 0)
        tn = summary.get("true_negatives", 0)
        fn = summary.get("false_negatives", 0)

        confusion = np.array([[tp, fn], [fp, tn]])

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(confusion, cmap='Blues', aspect='auto')

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Attacked', 'Clean'], fontsize=11)
        ax.set_yticklabels(['Detected', 'Not Detected'], fontsize=11)
        ax.set_xlabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')

        # Annotate cells
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, confusion[i, j],
                             ha="center", va="center",
                             color="white" if confusion[i, j] > confusion.max()/2 else "black",
                             fontsize=24, fontweight='bold')

        # Add metrics
        precision = summary.get("precision", 0)
        recall = summary.get("recall", 0)
        accuracy = summary.get("accuracy", 0)

        metrics_text = f"Precision: {precision:.2%}\nRecall: {recall:.2%}\nAccuracy: {accuracy:.2%}"
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f"Confusion Matrix — {get_title(data)}",
                    fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_feature_importance(pdf: PdfPages, data: dict):
        """Which MI features triggered most detections"""
        epochs = data.get("epochs", [])
        if not epochs:
            return

        threshold = data.get("metadata", {}).get("z_threshold", 2.5)

        # Count how many times each feature exceeded threshold
        trigger_counts = defaultdict(int)

        for e in epochs:
            if e.get("anomaly_detected"):
                z_scores = e.get("z_scores", {})
                for feat in MI_FEATURES:
                    z_key = f"{feat}_z"
                    if abs(z_scores.get(z_key, 0)) > threshold:
                        trigger_counts[feat] += 1

        if not trigger_counts:
            return

        # Sort by count
        sorted_features = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        features = [NICE_NAMES.get(f, f) for f, _ in sorted_features]
        counts = [c for _, c in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(features, counts, color=colors, edgecolor='black', linewidth=0.7)

        # Annotate bars
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{count}', ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Number of Detections Triggered', fontsize=11, fontweight='bold')
        ax.set_title(f"MI Feature Importance (Detection Triggers) — {get_title(data)}",
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(counts) * 1.15)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_entropy_comparison(pdf: PdfPages, data: dict):
        """Compare entropy values alongside MI"""
        epochs = sort_epochs_chronologically(data.get("epochs", []) or [])
        if not epochs:
            return

        baseline = data.get("baseline", {})
        indices = []
        for e in epochs:
            v = e.get("epoch_index")
            try:
                indices.append(int(v) if v is not None else len(indices))
            except (TypeError, ValueError):
                indices.append(len(indices))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()

        for idx, feat in enumerate(ENTROPY_FEATURES):
            ax = axes[idx]

            values = [e["mi_metrics"].get(feat, float('nan')) for e in epochs]
            baseline_mean = baseline.get(f"{feat}_mean", 0)
            baseline_std = baseline.get(f"{feat}_std", 0)

            for e in epochs:
                if epoch_ground_truth(e) is True:
                    xi = e.get("epoch_index")
                    try:
                        x0 = float(xi)
                    except (TypeError, ValueError):
                        continue
                    ax.axvspan(x0 - 0.5, x0 + 0.5, color="#F8B4B4", alpha=0.55, linewidth=0, zorder=0)

            # Baseline band
            ax.axhline(baseline_mean, color=CLR_BASELINE, linestyle='--', linewidth=1.5)
            ax.fill_between(indices,
                           baseline_mean - 2*baseline_std,
                           baseline_mean + 2*baseline_std,
                           color=CLR_BASELINE, alpha=0.15)

            # Observed
            ax.plot(indices, values, 'o-', color=CLR_OBSERVED, markersize=4, linewidth=1.2)

            ax.set_title(NICE_NAMES.get(feat, feat), fontsize=10, fontweight='bold')
            ax.set_ylabel('Entropy (bits)', fontsize=8)
            ax.grid(alpha=0.3)

            if idx >= 2:
                ax.set_xlabel('Epoch Index', fontsize=8)

        fig.suptitle(f"Entropy Features — {get_title(data)}",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def plot_multi_comparison(pdf: PdfPages, loaded: List[Tuple[str, dict]]):
        """Compare detection rates across multiple runs"""
        if len(loaded) < 2:
            return

        labels = []
        precisions = []
        recalls = []
        accuracies = []

        for path, data in loaded:
            if not is_sparse(data):
                continue

            summary = data.get("detection_summary", {})
            meta = data.get("metadata", {})

            label = f"{meta.get('baseline_percentage', '?')}%/{meta.get('malicious_percentage', '?')}%"
            labels.append(label)
            precisions.append(summary.get("precision", 0))
            recalls.append(summary.get("recall", 0))
            accuracies.append(summary.get("accuracy", 0))

        if not labels:
            return

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - width, precisions, width, label='Precision', color='#6699FF', edgecolor='black', linewidth=0.7)
        ax.bar(x, recalls, width, label='Recall', color='#FF9966', edgecolor='black', linewidth=0.7)
        ax.bar(x + width, accuracies, width, label='Accuracy', color='#66CC99', edgecolor='black', linewidth=0.7)

        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_xlabel('Configuration (Baseline%/Malicious%)', fontsize=11, fontweight='bold')
        ax.set_title('Detection Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for i, (p, r, a) in enumerate(zip(precisions, recalls, accuracies)):
            ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, a + 0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize MI Detector results with comprehensive plots"
    )
    parser.add_argument("inputs", nargs="+",
                       help="JSON file(s) or directory containing MI detector outputs")
    parser.add_argument("--reports-dir", default=None,
                       help="Batch report output folder (default: Mutual_Info/reports/ next to this script)")
    parser.add_argument("--batch-reports", action="store_true",
                       help="Generate one minimal PDF per JSON into reports-dir")
    parser.add_argument("--minimal", action="store_true",
                       help="Only generate minimal analysis pages (no full timelines/extra plots)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output PDF path (default: <first_input>_analysis.pdf)")
    parser.add_argument("--show", action="store_true",
                       help="Display plots interactively")
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        metavar="N",
        help="Epochs per row in attack/detection window plot and MI %% grouping (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="When no valid JSON is found, list every rejected file instead of first 5",
    )
    args = parser.parse_args()

    # Load all JSON files
    loaded = collect_jsons(args.inputs, verbose=args.verbose)
    if not loaded:
        print("No JSON files found.")
        sys.exit(1)

    print(f"\nLoaded {len(loaded)} file(s)")

    # Print text summaries
    for path, data in loaded:
        print_summary(data, os.path.basename(path))

    if not HAS_MPL:
        print("\n[PDF reports skipped — install matplotlib and numpy]")
        print("  pip install matplotlib numpy")
        return

    # Default reports dir: Mutual_Info/reports next to this script
    if args.reports_dir:
        reports_dir = Path(args.reports_dir)
    else:
        reports_dir = Path(__file__).resolve().parent / "reports"

    # Batch mode: one PDF per JSON into reports folder
    # Trigger if explicitly requested OR if multiple inputs and no --output provided.
    batch_mode = bool(args.batch_reports) or (args.output is None and len(loaded) > 1)
    if batch_mode:
        reports_dir.mkdir(parents=True, exist_ok=True)
        reports_abs = str(reports_dir.resolve())
        print(f"\nGenerating per-JSON reports → {reports_abs}/")
        for path, data in loaded:
            stem = Path(path).stem
            out_path = reports_dir / f"{stem}.pdf"
            label = os.path.basename(path)
            with PdfPages(str(out_path)) as pdf:
                generate_minimal_report(pdf, data, label, group_size=args.window_size)
                if not args.minimal:
                    # Keep extra pages modest: only high-signal extras
                    plot_detection_overview(pdf, data)
                    plot_feature_importance(pdf, data)
            print(f"  ✓ {out_path.resolve()}")
        print("\n✓ Batch complete.")
        return

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        first_path = loaded[0][0]
        if os.path.isfile(first_path):
            stem = Path(first_path).stem
            parent = Path(first_path).parent
        else:
            stem = os.path.basename(first_path.rstrip("/"))
            parent = Path(first_path)
        out_path = str(parent / f"{stem}_mi_analysis.pdf")

    out_abs = str(Path(out_path).resolve())
    print(f"\nGenerating plots → {out_abs}")

    if args.show:
        matplotlib.use("TkAgg")

    # Generate plots
    with PdfPages(out_path) as pdf:
        for path, data in loaded:
            print(f"  Plotting: {os.path.basename(path)}")
            generate_minimal_report(pdf, data, os.path.basename(path), group_size=args.window_size)
            if not args.minimal:
                plot_mi_timelines(pdf, data)
                plot_z_scores(pdf, data)
                plot_detection_overview(pdf, data)
                plot_confusion_matrix(pdf, data)
                plot_feature_importance(pdf, data)
                plot_entropy_comparison(pdf, data)

        # Multi-file comparison
        if len(loaded) > 1 and not args.minimal:
            print("  Plotting: Multi-file comparison")
            plot_multi_comparison(pdf, loaded)

    print(f"\n✓ Complete! → {out_abs}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
