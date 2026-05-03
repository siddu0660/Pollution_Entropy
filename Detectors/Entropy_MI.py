"""
Entropy_MI.py — Analysis of Entropy_Stream.py JSON outputs.

Accepts one or more JSON files (or a directory containing them) and produces:
  1. Mutual Information ranking bar chart (if MI key present).
  2. Per-epoch timeline: clean vs polluted metrics for every feature.
  3. Entropy-change distributions (box-plots) grouped by feature.
  4. (Sparse mode) attacked vs unattacked epoch comparison plots.
  5. Printed summary table.

Saves all plots to a multi-page PDF next to the first input JSON, or to a
path supplied via --output.

Usage:
    python Entropy_MI.py <json_or_dir> [<json_or_dir> ...] [--output out.pdf] [--show]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── optional import guard ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found — only text summary will be produced.")

# ── constants ────────────────────────────────────────────────────────────────
ENTROPY_KEYS    = ["src_ip_entropy", "dst_ip_entropy", "src_port_entropy", "dst_port_entropy"]
CARDINALITY_KEYS= ["src_ip_cardinality", "dst_ip_cardinality",
                   "src_port_cardinality", "dst_port_cardinality"]
UNIFORMITY_KEYS = ["src_ip_uniformity", "dst_ip_uniformity"]
RATIO_KEYS      = ["src_dst_ratio"]

ALL_FEATURE_KEYS = ENTROPY_KEYS + CARDINALITY_KEYS + UNIFORMITY_KEYS + RATIO_KEYS

CHANGE_KEYS = [k + "_change" for k in
               ["src_ip_entropy", "dst_ip_entropy", "src_port_entropy", "dst_port_entropy",
                "src_ip_cardinality", "dst_ip_cardinality",
                "src_port_cardinality", "dst_port_cardinality",
                "src_ip_uniformity", "dst_ip_uniformity",
                "src_dst_ratio"]]

NICE_NAME = {
    "src_ip_entropy":          "Src-IP Entropy",
    "dst_ip_entropy":          "Dst-IP Entropy",
    "src_port_entropy":        "Src-Port Entropy",
    "dst_port_entropy":        "Dst-Port Entropy",
    "src_ip_cardinality":      "Src-IP Cardinality",
    "dst_ip_cardinality":      "Dst-IP Cardinality",
    "src_port_cardinality":    "Src-Port Cardinality",
    "dst_port_cardinality":    "Dst-Port Cardinality",
    "src_ip_uniformity":       "Src-IP Uniformity",
    "dst_ip_uniformity":       "Dst-IP Uniformity",
    "src_dst_ratio":           "Src/Dst Ratio",
}

CLR_CLEAN   = "#4C88C8"
CLR_POLLUTED= "#E05A3A"
CLR_ATK     = "#E05A3A"
CLR_CLEAN_S = "#4C88C8"

# ─────────────────────────────── helpers ────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_jsons(paths: List[str]) -> List[Tuple[str, dict]]:
    """Expand directories and load all JSON files."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            for fn in sorted(Path(p).glob("*.json")):
                result.append((str(fn), load_json(str(fn))))
        else:
            result.append((p, load_json(p)))
    return result


def is_sparse(data: dict) -> bool:
    return data.get("metadata", {}).get("sparse_mode", False)


def epoch_series(epochs: List[dict], key: str, source: str = "malicious_metrics") -> List[float]:
    """Extract a list of values from epochs[*][source][key]."""
    return [e.get(source, {}).get(key, float("nan")) for e in epochs]


def epoch_attacked(epochs: List[dict]) -> List[bool]:
    return [e.get("attacked", True) for e in epochs]


def stats(values: List[float]) -> Dict[str, float]:
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return {}
    mean  = sum(v) / len(v)
    var   = sum((x - mean)**2 for x in v) / len(v)
    std   = math.sqrt(var)
    return {"mean": mean, "std": std, "min": min(v), "max": max(v), "n": len(v)}


# ─────────────────────────── text summary ───────────────────────────────────

def print_summary(data: dict, label: str) -> None:
    meta = data.get("metadata", {})
    epochs = data.get("epochs", [])
    mi = data.get("mutual_information", {})

    print("\n" + "="*72)
    print(f"  FILE : {label}")
    print(f"  Dataset : {meta.get('dataset_name','?')}")
    print(f"  Baseline: {meta.get('baseline_percentage','?')}%  "
          f"Malicious: {meta.get('malicious_percentage','?')}%  "
          f"Sparse: {meta.get('sparse_mode', False)}")
    print(f"  Total epochs (non-baseline): {len(epochs)}")

    attacked     = [e for e in epochs if e.get("attacked", True)]
    not_attacked = [e for e in epochs if not e.get("attacked", True)]
    print(f"  Attacked epochs: {len(attacked)}   Unattacked: {len(not_attacked)}")

    # Entropy changes summary
    print("\n  --- Entropy changes (malicious vs baseline) ---")
    print(f"  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-"*66)
    for ck in CHANGE_KEYS:
        values = [e.get("entropy_changes", {}).get(ck, float("nan"))
                  for e in attacked if e.get("attacked", True)]
        s = stats(values)
        if s:
            name = NICE_NAME.get(ck.replace("_change", ""), ck)
            print(f"  {name:<30} {s['mean']:>10.4f} {s['std']:>10.4f} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f}")

    # MI summary
    if mi:
        print("\n  --- Mutual Information (bits, feature vs attack label) ---")
        sorted_mi = sorted(mi.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_mi:
            bar = "█" * int(score * 40 / max(mi.values(), default=1))
            name = NICE_NAME.get(feat, feat)
            print(f"  {name:<30} {score:.6f}  {bar}")
    else:
        print("\n  [No mutual_information key — run with updated Entropy_Stream.py]")

    print("="*72)


# ─────────────────────────── matplotlib plots ───────────────────────────────
if HAS_MPL:
    def _fig_title(data: dict) -> str:
        m = data.get("metadata", {})
        return (f"{m.get('dataset_name','?')}  |  "
                f"Baseline {m.get('baseline_percentage','?')}%  "
                f"Mal {m.get('malicious_percentage','?')}%"
                + ("  [SPARSE]" if m.get("sparse_mode") else ""))

    # 1. MI bar chart
    def plot_mi(pdf: "PdfPages", data: dict) -> None:
        mi = data.get("mutual_information")
        if not mi:
            return
        sorted_mi = sorted(mi.items(), key=lambda x: x[1])
        labels = [NICE_NAME.get(k, k) for k, _ in sorted_mi]
        values = [v for _, v in sorted_mi]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#4C88C8" if v >= 0.5 * max(values, default=1) else "#A0BDD8" for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
        ax.set_xlabel("Mutual Information (bits)", fontsize=11)
        ax.set_title(f"Feature Mutual Information — {_fig_title(data)}", fontsize=11, fontweight="bold")
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlim(left=0, right=max(values) * 1.25 if values else 1)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # 2. Per-epoch timeline (clean vs polluted)
    def plot_epoch_timelines(pdf: "PdfPages", data: dict, feature_group: List[str], group_name: str) -> None:
        epochs = data.get("epochs", [])
        if not epochs:
            return
        indices = [e["epoch_index"] for e in epochs]
        attacked_mask = epoch_attacked(epochs)

        ncols = 2
        nrows = math.ceil(len(feature_group) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax_idx, key in enumerate(feature_group):
            ax = axes[ax_idx]
            clean_vals    = epoch_series(epochs, key, "clean_metrics")
            polluted_vals = epoch_series(epochs, key, "malicious_metrics")

            if is_sparse(data):
                # shade attacked vs not-attacked epochs
                for i, (ix, atk) in enumerate(zip(indices, attacked_mask)):
                    c = CLR_ATK if atk else CLR_CLEAN_S
                    ax.axvspan(ix - 0.5, ix + 0.5, color=c, alpha=0.10, linewidth=0)

            ax.plot(indices, clean_vals,    color=CLR_CLEAN,    linewidth=1.2, label="Clean")
            ax.plot(indices, polluted_vals, color=CLR_POLLUTED, linewidth=1.2, linestyle="--", label="Polluted")
            ax.set_title(NICE_NAME.get(key, key), fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)
            if ax_idx % ncols == 0:
                ax.set_ylabel("Value", fontsize=8)

        # legend
        handles = [
            mpatches.Patch(color=CLR_CLEAN, label="Clean"),
            mpatches.Patch(color=CLR_POLLUTED, label="Polluted / Injected"),
        ]
        if is_sparse(data):
            handles += [
                mpatches.Patch(color=CLR_ATK,    alpha=0.3, label="Attacked epoch"),
                mpatches.Patch(color=CLR_CLEAN_S, alpha=0.3, label="Unattacked epoch"),
            ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"{group_name}  —  {_fig_title(data)}", fontsize=11, fontweight="bold")

        # hide unused axes
        for ax in axes[len(feature_group):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        pdf.savefig(fig)
        plt.close(fig)

    # 3. Entropy-change boxplots (attacked epochs only)
    def plot_change_boxplots(pdf: "PdfPages", data: dict) -> None:
        epochs = data.get("epochs", [])
        attacked = [e for e in epochs if e.get("attacked", True)]
        if not attacked:
            return

        change_data = []
        change_labels = []
        for ck in CHANGE_KEYS:
            vals = [e.get("entropy_changes", {}).get(ck, float("nan")) for e in attacked]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                change_data.append(vals)
                change_labels.append(NICE_NAME.get(ck.replace("_change", ""), ck))

        if not change_data:
            return

        fig, ax = plt.subplots(figsize=(13, 6))
        bp = ax.boxplot(change_data, vert=True, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        boxprops=dict(facecolor="#4C88C8", alpha=0.7))
        ax.set_xticks(range(1, len(change_labels) + 1))
        ax.set_xticklabels(change_labels, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Change vs Baseline", fontsize=10)
        ax.set_title(f"Entropy Change Distribution (attacked epochs) — {_fig_title(data)}",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # 4. Sparse: per-feature mean of attacked vs unattacked
    def plot_sparse_comparison(pdf: "PdfPages", data: dict) -> None:
        if not is_sparse(data):
            return
        epochs = data.get("epochs", [])
        attacked     = [e for e in epochs if e.get("attacked", True)]
        not_attacked = [e for e in epochs if not e.get("attacked", True)]
        if not attacked or not not_attacked:
            return

        features = ENTROPY_KEYS + CARDINALITY_KEYS
        atk_means  = []
        noatk_means= []
        for k in features:
            av = [e["malicious_metrics"].get(k, float("nan")) for e in attacked
                  if "malicious_metrics" in e]
            nav= [e["malicious_metrics"].get(k, float("nan")) for e in not_attacked
                  if "malicious_metrics" in e]
            av  = [x for x in av  if not math.isnan(x)]
            nav = [x for x in nav if not math.isnan(x)]
            atk_means.append(sum(av)  / len(av)  if av  else 0)
            noatk_means.append(sum(nav)/ len(nav) if nav else 0)

        labels = [NICE_NAME.get(k, k) for k in features]
        x = list(range(len(features)))
        w = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar([xi - w/2 for xi in x], atk_means,  width=w, color=CLR_ATK,    label="Attacked epochs", alpha=0.85)
        ax.bar([xi + w/2 for xi in x], noatk_means, width=w, color=CLR_CLEAN_S, label="Unattacked epochs", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean observed value", fontsize=10)
        ax.set_title(f"Attacked vs Unattacked Epoch Means (sparse) — {_fig_title(data)}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # 5. Combined MI comparison across multiple files
    def plot_multi_mi(pdf: "PdfPages", loaded: List[Tuple[str, dict]]) -> None:
        datasets = [(os.path.basename(p), d) for p, d in loaded if d.get("mutual_information")]
        if len(datasets) < 2:
            return
        features = ALL_FEATURE_KEYS
        labels_f = [NICE_NAME.get(k, k) for k in features]
        x = list(range(len(features)))
        w = 0.8 / len(datasets)

        fig, ax = plt.subplots(figsize=(14, 6))
        cmap = plt.cm.get_cmap("tab10", len(datasets))
        for i, (label, d) in enumerate(datasets):
            mi = d["mutual_information"]
            vals = [mi.get(k, 0.0) for k in features]
            offsets = [xi + (i - len(datasets)/2 + 0.5) * w for xi in x]
            ax.bar(offsets, vals, width=w * 0.9, color=cmap(i), alpha=0.85, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_f, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("MI (bits)", fontsize=10)
        ax.set_title("Mutual Information Comparison — All Files", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────────────

def default_output(paths: List[str]) -> str:
    first = paths[0]
    stem  = Path(first).stem if os.path.isfile(first) else os.path.basename(first.rstrip("/"))
    parent= Path(first).parent if os.path.isfile(first) else Path(first)
    return str(parent / f"{stem}_mi_analysis.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse Entropy_Stream JSON outputs (MI, timelines, boxplots)."
    )
    parser.add_argument("inputs", nargs="+",
                        help="JSON file(s) or director(y/ies) containing JSON files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF path (default: <first_input>_mi_analysis.pdf)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (requires a display)")
    args = parser.parse_args()

    loaded = collect_jsons(args.inputs)
    if not loaded:
        print("No JSON files found.")
        sys.exit(1)

    print(f"\nLoaded {len(loaded)} JSON file(s).")

    # Text summary for all files
    for path, data in loaded:
        print_summary(data, os.path.basename(path))

    if not HAS_MPL:
        print("\n[Skipping PDF generation — install matplotlib to enable plots.]")
        return

    out_path = args.output or default_output([p for p, _ in loaded])
    print(f"\nWriting PDF → {out_path}")

    if args.show:
        matplotlib.use("TkAgg")

    with PdfPages(out_path) as pdf:
        # Per-file plots
        for path, data in loaded:
            plot_mi(pdf, data)
            plot_epoch_timelines(pdf, data, ENTROPY_KEYS,     "Entropy Features")
            plot_epoch_timelines(pdf, data, CARDINALITY_KEYS, "Cardinality Features")
            plot_epoch_timelines(pdf, data, UNIFORMITY_KEYS + RATIO_KEYS, "Uniformity & Ratio")
            plot_change_boxplots(pdf, data)
            plot_sparse_comparison(pdf, data)

        # Cross-file MI comparison
        if HAS_MPL and len(loaded) > 1:
            plot_multi_mi(pdf, loaded)

    print(f"Done — {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
