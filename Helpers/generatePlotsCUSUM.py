"""
generatePlotsCUSUM.py  –  PDF Plot Generator for Entropy_CUSUM.py Output
=========================================================================

Usage:
    python generatePlotsCUSUM.py <json_folder>

Produces one PDF per JSON file, saved to <json_folder>/../results/.

Pages per PDF
-------------
  1. Cover / summary page  (metadata, baseline CUSUM stats, detection summary)
  2-4. Per-signal pages    (one page each for: js, new_src_ratio, topk_src)
         Each page has 3 sub-panels:
           • Raw signal value vs epoch
           • C_plus  (upward CUSUM) vs epoch, with alarm threshold h
           • C_minus (downward CUSUM) vs epoch, with alarm threshold h
  5. Alarm timeline heatmap  (epoch × signal grid, red = alarm fired)
  6. Detection performance table  (TP / FP / TN / FN / Precision / Recall / F1)
"""

import json
import sys
import os
import math
import glob
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E8E8E8",
    "grid.linewidth":     0.6,
    "axes.edgecolor":     "#CCCCCC",
    "axes.linewidth":     0.8,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.labelcolor":    "#333333",
    "axes.labelsize":     9,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# Signal definitions: (json_key, display_label, accent_color)
SIGNALS = [
    ("js",            "Combined JS Divergence",       "#2563EB"),   # blue
    ("new_src_ratio", "New Source IP Ratio",           "#16A34A"),   # green
    ("topk_src",      "Top-5 Source Concentration",   "#D97706"),   # amber
]

ALARM_COLOR   = "#DC2626"   # red   — alarm or attacked epoch
CLEAN_COLOR   = "#2563EB"   # blue  — clean epoch
CPLUS_COLOR   = "#DC2626"   # red   — C+
CMINUS_COLOR  = "#7C3AED"   # violet — C-


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xtick_step(n: int) -> int:
    """Return a sensible x-tick stride so we never show > ~15 labels."""
    return max(1, math.ceil(n / 14))


def _epoch_label(ep: Dict) -> str:
    return ep.get("epoch_name", f"E{ep['epoch_index'] + 1}")


def _format_val(v: float) -> str:
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    elif abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.2e}"


# ---------------------------------------------------------------------------
# Page 1 — Cover / Summary
# ---------------------------------------------------------------------------

def draw_cover(pdf: PdfPages, data: Dict, json_name: str, det: Dict):
    metadata  = data.get("metadata", {})
    baseline  = data.get("baseline", {})
    epochs    = data.get("epochs", [])

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.96, "CUSUM Change-Point Detection Analysis",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color="#111111")
    ax.text(0.5, 0.922, "Cumulative Sum Control Chart  |  JS Divergence + New IP Ratio + Top-K Source",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11, color="#666666")
    ax.plot([0.06, 0.94], [0.905, 0.905],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)

    # ── Run parameters (left) ────────────────────────────────────────────────
    is_sparse = metadata.get("sparse_mode", False)
    params = [
        ("Dataset",           json_name),
        ("Timestamp",         metadata.get("timestamp", "—")),
        ("Method",            metadata.get("method", "CUSUM")),
        ("Total epochs",      str(metadata.get("total_epochs", "—"))),
        ("Baseline epochs",   (f"{metadata.get('baseline_epoch_count', '—')}"
                               f"  ({metadata.get('baseline_percentage', '?')}%)")),
        ("Attack epochs",     str(len(epochs))),
        ("Malicious traffic", f"{metadata.get('malicious_percentage', '?')}%"),
        ("CUSUM k-factor",    str(metadata.get("k_factor", "—"))),
        ("CUSUM h-factor",    str(metadata.get("h_factor", "—"))),
        ("Mode",              "SPARSE" if is_sparse else "FULL"),
    ]
    if is_sparse:
        aw = metadata.get("attacked_windows", [])
        params += [
            ("Attack probability", str(metadata.get("sparse_attack_prob", "—"))),
            ("Window size",        str(metadata.get("sparse_window_size", "—"))),
            ("Windows attacked",   str(len(aw))),
        ]

    ax.text(0.06, 0.880, "Run Parameters",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#333333")
    y = 0.850
    for k, v in params:
        ax.text(0.06, y, k, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#666666")
        ax.text(0.28, y, v, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#111111")
        y -= 0.040

    # ── Baseline CUSUM stats (middle) ────────────────────────────────────────
    ax.text(0.42, 0.880, "Baseline CUSUM Parameters",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#333333")

    sig_labels = {"js": "Combined JS", "new_src_ratio": "New-IP Ratio", "topk_src": "Top-K Src"}
    y = 0.850
    for sig_key, sig_lbl in sig_labels.items():
        bline = baseline.get(sig_key, {})
        row = (f"μ={bline.get('mean', 0):.4f}  "
               f"σ={bline.get('std', 0):.4f}  "
               f"k={bline.get('k', 0):.4f}  "
               f"h={bline.get('h', 0):.4f}")
        ax.text(0.42, y, sig_lbl, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#666666")
        ax.text(0.57, y, row, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#111111")
        y -= 0.040

    # ── Detection summary (right) ─────────────────────────────────────────────
    ax.text(0.70, 0.880, "Detection Summary (Any Signal)",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#333333")
    det_rows = [
        ("True Positives (TP)",  str(det["tp"])),
        ("False Positives (FP)", str(det["fp"])),
        ("True Negatives (TN)",  str(det["tn"])),
        ("False Negatives (FN)", str(det["fn"])),
        ("Precision",            f"{det['precision']:.3f}"),
        ("Recall",               f"{det['recall']:.3f}"),
        ("F1 Score",             f"{det['f1']:.3f}"),
        ("Detection Rate",       f"{det['detection_rate'] * 100:.1f}%"),
    ]
    y = 0.850
    for k, v in det_rows:
        ax.text(0.70, y, k, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#666666")
        ax.text(0.88, y, v, ha="left", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#111111", fontweight="bold")
        y -= 0.040

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.plot([0.06, 0.94], [0.055, 0.055],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.5, 0.040,
            "Subsequent pages: per-signal 3-panel plots (raw signal, C+, C−), "
            "alarm timeline heatmap, and detection performance table.",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7.5, color="#999999", fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 2-4 — Per-signal 3-panel plot
# ---------------------------------------------------------------------------

def draw_signal_page(pdf: PdfPages, sig_key: str, sig_label: str,
                     sig_color: str, epochs: List[Dict],
                     baseline_h: float, baseline_mean: float,
                     malicious_pct, baseline_pct, is_sparse: bool,
                     attacked_set: set):
    """
    Three stacked panels:
      Top    – raw signal value per epoch
      Middle – C_plus  per epoch with threshold h
      Bottom – C_minus per epoch with threshold h
    """
    n = len(epochs)
    if n == 0:
        return

    xs            = list(range(n))
    raw_vals      = [ep["signal"].get(sig_key, 0.0) for ep in epochs]
    c_plus_vals   = [ep["cusum"].get(sig_key, {}).get("c_plus", 0.0) for ep in epochs]
    c_minus_vals  = [ep["cusum"].get(sig_key, {}).get("c_minus", 0.0) for ep in epochs]
    alarm_vals    = [ep["cusum"].get(sig_key, {}).get("alarm", False) for ep in epochs]
    attacked_bool = [ep.get("attacked", False) for ep in epochs]

    labels = [_epoch_label(ep) for ep in epochs]
    step   = _xtick_step(n)
    shown  = xs[::step]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5),
                             gridspec_kw={"hspace": 0.52})
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.08)

    panel_specs = [
        (axes[0], raw_vals,     "Raw Signal Value",  sig_color,   False),
        (axes[1], c_plus_vals,  "C\u207a (Upward CUSUM)",    CPLUS_COLOR, True),
        (axes[2], c_minus_vals, "C\u207b (Downward CUSUM)", CMINUS_COLOR, True),
    ]

    for ax, vals, ylabel, color, show_h in panel_specs:
        ax.plot(xs, vals, color=color, linewidth=1.0, alpha=0.5, zorder=2)

        # Scatter: alarm = red circle, clean = small grey
        for xi, (v, alarm, atk) in enumerate(zip(vals, alarm_vals, attacked_bool)):
            if alarm:
                ax.scatter(xi, v, color=ALARM_COLOR, s=45, zorder=5,
                           edgecolors="white", linewidths=0.5)
            else:
                c = "#DC262622" if atk else "#33333311"
                ax.scatter(xi, v, color=color, s=14, zorder=3,
                           edgecolors="none", alpha=0.7)

        # Threshold line
        if show_h and baseline_h > 0:
            ax.axhline(baseline_h, color="#DC262655", linewidth=0.9,
                       linestyle="--", label=f"h = {baseline_h:.4f}")

        # Baseline mean on raw panel
        if not show_h:
            ax.axhline(baseline_mean, color="#33333355", linewidth=0.8,
                       linestyle=":", label=f"baseline μ = {baseline_mean:.4f}")

        # Shade attacked epochs
        if attacked_set:
            for xi, atk in enumerate(attacked_bool):
                if atk:
                    ax.axvspan(xi - 0.5, xi + 0.5, color=ALARM_COLOR,
                               alpha=0.06, zorder=0)

        ax.set_xlim(-0.8, n - 0.2)
        ax.set_xticks(shown)
        ax.set_xticklabels([labels[i] for i in shown],
                            rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))
        ax.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.3)

    # Titles
    fig.text(0.5, 0.93, sig_label,
             ha="center", va="bottom",
             fontsize=15, fontweight="bold", color="#111111")
    fig.text(0.5, 0.902, f"CUSUM Signal Analysis  |  malicious = {malicious_pct}%  |  baseline = {baseline_pct}%",
             ha="center", va="bottom",
             fontsize=9, color="#666666")

    # Legend handles
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ALARM_COLOR,
               markersize=7, label="CUSUM alarm fired"),
        mpatches.Patch(facecolor=ALARM_COLOR, alpha=0.10, label="Attacked epoch"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8,
               loc="upper right", bbox_to_anchor=(0.97, 0.975))

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5 — Alarm timeline heatmap
# ---------------------------------------------------------------------------

def draw_alarm_heatmap(pdf: PdfPages, epochs: List[Dict]):
    n = len(epochs)
    if n == 0:
        return

    signal_keys  = [s[0] for s in SIGNALS]
    signal_names = [s[1] for s in SIGNALS]
    n_sigs       = len(signal_keys)

    # Build matrix: rows = signals, cols = epochs
    matrix = np.zeros((n_sigs, n), dtype=float)
    for ei, ep in enumerate(epochs):
        for si, sk in enumerate(signal_keys):
            if ep["cusum"].get(sk, {}).get("alarm", False):
                matrix[si, ei] = 1.0

    # Overall alarm row
    overall = np.array([1.0 if ep.get("cusum_alarm", False) else 0.0
                        for ep in epochs])
    attacked = np.array([1.0 if ep.get("attacked", False) else 0.0
                         for ep in epochs])

    n_rows = n_sigs + 2  # +attacked +overall
    full_matrix = np.vstack([attacked, matrix, overall])
    row_labels  = ["Attacked (Ground Truth)"] + signal_names + ["Any-Signal Alarm"]

    step  = _xtick_step(n)
    shown = list(range(0, n, step))
    xlabels = [_epoch_label(epochs[i]) for i in shown]

    fig, ax = plt.subplots(figsize=(11, 1.0 + n_rows * 0.65))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.15)

    # Custom colour map: white→color
    from matplotlib.colors import ListedColormap
    colors_per_row = [
        "#DC2626",   # attacked — red
        SIGNALS[0][2],
        SIGNALS[1][2],
        SIGNALS[2][2],
        "#111111",   # overall alarm — dark
    ]

    for ri in range(n_rows):
        for ci in range(n):
            val = full_matrix[ri, ci]
            fc  = colors_per_row[ri] if val > 0 else "#F3F4F6"
            rect = mpatches.FancyBboxPatch(
                (ci - 0.45, ri - 0.40), 0.90, 0.80,
                boxstyle="round,pad=0.02", linewidth=0,
                facecolor=fc, edgecolor="none")
            ax.add_patch(rect)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(shown)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7)
    ax.set_xlabel("Epoch", labelpad=5, fontsize=9)
    ax.invert_yaxis()
    ax.grid(False)

    fig.text(0.5, 0.94, "CUSUM Alarm Timeline",
             ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="#111111")
    fig.text(0.5, 0.91, "Filled cell = alarm fired for that epoch × signal",
             ha="center", va="bottom",
             fontsize=9, color="#666666")

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=r)
               for c, r in zip(colors_per_row, row_labels)]
    handles.append(mpatches.Patch(facecolor="#F3F4F6", edgecolor="#CCCCCC",
                                  label="No alarm"))
    ax.legend(handles=handles, frameon=False, fontsize=7,
              loc="lower right", ncol=2)

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 6 — Detection performance table
# ---------------------------------------------------------------------------

def draw_detection_table(pdf: PdfPages, det: Dict, metadata: Dict):
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")

    fig.text(0.5, 0.93, "Detection Performance Metrics",
             ha="center", va="bottom",
             fontsize=16, fontweight="bold", color="#111111")
    fig.text(0.5, 0.87,
             f"Dataset: {metadata.get('dataset_name', '—')}   |   "
             f"Malicious: {metadata.get('malicious_percentage', '?')}%   |   "
             f"Baseline: {metadata.get('baseline_percentage', '?')}%",
             ha="center", va="bottom", fontsize=10, color="#666666")

    # Table data
    table_data = [
        ["Metric",          "Value",                           "Description"],
        ["True Positives",  str(det["tp"]),                    "Attacked epochs correctly alarmed"],
        ["False Positives", str(det["fp"]),                    "Clean epochs incorrectly alarmed"],
        ["True Negatives",  str(det["tn"]),                    "Clean epochs correctly not alarmed"],
        ["False Negatives", str(det["fn"]),                    "Attacked epochs missed"],
        ["Precision",       f"{det['precision']:.4f}",         "TP / (TP + FP)"],
        ["Recall",          f"{det['recall']:.4f}",            "TP / (TP + FN)"],
        ["F1 Score",        f"{det['f1']:.4f}",                "Harmonic mean of Precision & Recall"],
        ["Detection Rate",  f"{det['detection_rate']*100:.2f}%", "Fraction of attacked epochs detected"],
        ["False Alarm Rate",f"{det['false_alarm_rate']*100:.2f}%","Fraction of clean epochs with alarm"],
    ]

    n_rows = len(table_data)
    n_cols = 3
    col_xs = [0.08, 0.32, 0.58]
    row_h  = 0.063
    y0     = 0.80

    for ri, row in enumerate(table_data):
        y = y0 - ri * row_h
        bg = "#F8FAFC" if ri % 2 == 0 else "white"
        header = ri == 0
        rect = mpatches.FancyBboxPatch(
            (0.06, y - 0.01), 0.88, row_h - 0.005,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor="#1E3A5F" if header else bg,
            transform=ax.transAxes)
        ax.add_patch(rect)
        for ci, (cell, cx) in enumerate(zip(row, col_xs)):
            ax.text(cx, y + row_h * 0.28, cell,
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=9 if not header else 9,
                    fontweight="bold" if header or ci == 0 else "normal",
                    color="white" if header else "#111111")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detection metric calculation
# ---------------------------------------------------------------------------

def compute_detection_metrics(epochs: List[Dict]) -> Dict:
    tp = fp = tn = fn = 0
    for ep in epochs:
        alarm   = ep.get("cusum_alarm", False)
        attacked = ep.get("attacked", False)
        if attacked and alarm:
            tp += 1
        elif attacked and not alarm:
            fn += 1
        elif not attacked and alarm:
            fp += 1
        else:
            tn += 1

    precision      = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall         = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1             = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
    detection_rate = recall
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1,
                detection_rate=detection_rate,
                false_alarm_rate=false_alarm_rate)


# ---------------------------------------------------------------------------
# Process one JSON
# ---------------------------------------------------------------------------

def process_single_json(json_path: str, output_dir: str) -> bool:
    try:
        with open(json_path) as f:
            data: Dict[str, Any] = json.load(f)

        metadata  = data.get("metadata", {})
        baseline  = data.get("baseline", {})
        epochs    = data.get("epochs", [])

        if not epochs:
            print(f"  [WARNING] No epoch data in {os.path.basename(json_path)}, skipping.")
            return False

        json_stem = os.path.splitext(os.path.basename(json_path))[0]
        pdf_path  = os.path.join(output_dir, f"{json_stem}.pdf")

        mal_pct   = metadata.get("malicious_percentage", "0")
        base_pct  = metadata.get("baseline_percentage", "5")
        is_sparse = metadata.get("sparse_mode", False)

        # Build the set of epochs that were actually attacked
        attacked_set = {ep["epoch_index"] for ep in epochs if ep.get("attacked", False)}

        # Detection metrics
        det = compute_detection_metrics(epochs)

        with PdfPages(pdf_path) as pdf:
            # Page 1 — cover
            draw_cover(pdf, data, os.path.basename(json_path), det)

            # Pages 2-4 — per signal
            for sig_key, sig_label, sig_color in SIGNALS:
                bl = baseline.get(sig_key, {})
                draw_signal_page(
                    pdf         = pdf,
                    sig_key     = sig_key,
                    sig_label   = sig_label,
                    sig_color   = sig_color,
                    epochs      = epochs,
                    baseline_h  = bl.get("h", 0.0),
                    baseline_mean = bl.get("mean", 0.0),
                    malicious_pct = mal_pct,
                    baseline_pct  = base_pct,
                    is_sparse   = is_sparse,
                    attacked_set = attacked_set,
                )

            # Page 5 — alarm heatmap
            draw_alarm_heatmap(pdf, epochs)

            # Page 6 — detection table
            draw_detection_table(pdf, det, metadata)

            info = pdf.infodict()
            info["Title"]   = f"CUSUM Analysis — {json_stem}"
            info["Subject"] = f"Baseline {base_pct}% | Malicious {mal_pct}%"

        print(f"  [✓] {json_stem}.pdf")
        return True

    except Exception as e:
        import traceback
        print(f"  [ERROR] Failed to process {os.path.basename(json_path)}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        json_folder = sys.argv[1]
    else:
        json_folder = input("Enter the path to the json folder: ").strip()

    if not os.path.isdir(json_folder):
        print(f"[ERROR] Directory not found: {json_folder}")
        sys.exit(1)

    parent_dir  = os.path.dirname(os.path.abspath(json_folder))
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(json_folder, "*_cusum*.json")))
    if not json_files:
        # Fall back to all JSON files if no cusum-specific ones found
        json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    if not json_files:
        print(f"[ERROR] No JSON files found in: {json_folder}")
        sys.exit(1)

    print("=" * 80)
    print("CUSUM PLOT GENERATOR — BATCH MODE")
    print("=" * 80)
    print(f"\nJSON folder    : {json_folder}")
    print(f"Results folder : {results_dir}")
    print(f"JSON files     : {len(json_files)}")
    print("-" * 80)

    successful, failed = 0, 0
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {os.path.basename(json_path)}")
        if process_single_json(json_path, results_dir):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print(f"Successful : {successful}/{len(json_files)}")
    if failed:
        print(f"Failed     : {failed}/{len(json_files)}")
    print(f"PDFs saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
