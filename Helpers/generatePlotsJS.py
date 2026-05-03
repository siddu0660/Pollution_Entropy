import json, sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Any

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
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelcolor":    "#333333",
    "axes.labelsize":     10,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

CATEGORY_COLOR = {
    "JS Divergence":  "#2563EB",   # blue
    "KL Divergence":  "#DC2626",   # red
    "Combined":       "#16A34A",   # green
    "Flow":           "#D97706",   # amber
    "Count":          "#7C3AED",   # violet
}

GROUP_SIZE = 10

# ---------------------------------------------------------------------------
# Metric definitions  (key, label, category)
#   key      – key inside epoch["divergence_vs_baseline"]
#   label    – human-readable title for the plot page
#   category – colour bucket
# ---------------------------------------------------------------------------
METRICS = [
    # JS divergence — attacked epoch vs baseline
    ("src_ip_js",   "Source IP — JS Divergence",          "JS Divergence"),
    ("dst_ip_js",   "Destination IP — JS Divergence",     "JS Divergence"),
    ("src_port_js", "Source Port — JS Divergence",        "JS Divergence"),
    ("dst_port_js", "Destination Port — JS Divergence",   "JS Divergence"),
    ("combined_js", "Combined (all features) — JS",       "Combined"),
    # KL divergence — attacked epoch vs baseline
    ("src_ip_kl",   "Source IP — KL Divergence",          "KL Divergence"),
    ("dst_ip_kl",   "Destination IP — KL Divergence",     "KL Divergence"),
    ("src_port_kl", "Source Port — KL Divergence",        "KL Divergence"),
    ("dst_port_kl", "Destination Port — KL Divergence",   "KL Divergence"),
    # Flow count
    ("total_flows", "Total Flows",                        "Flow"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_dev(delta: float, base: float) -> float:
    """% deviation: (delta / base) * 100, or raw delta if base is near zero."""
    if base and abs(base) > 1e-12:
        return (delta / base) * 100.0
    return delta  # for JS/KL the base is already [0,1]; use raw delta × 100 looks right

def group_avg(vals: List[float], n: int) -> List[float]:
    return [float(np.mean(vals[i:i+n])) for i in range(0, len(vals), n)]

def group_labels(indices: List[int], n: int) -> List[str]:
    out = []
    for i in range(0, len(indices), n):
        chunk = indices[i:i+n]
        out.append(f"E{chunk[0]+1}–{chunk[-1]+1}")
    return out


# ---------------------------------------------------------------------------
# Summary / cover page
# ---------------------------------------------------------------------------

def draw_summary(pdf: PdfPages, metadata: dict, baseline: dict,
                 n_epochs: int, json_name: str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "JS / KL Divergence Analysis",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="#111111")
    ax.text(0.5, 0.905, "Epoch-wise Divergence from Baseline Distribution",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=12, color="#666666")
    ax.plot([0.08, 0.92], [0.885, 0.885],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)

    # Run parameters (left column)
    is_sparse = metadata.get("sparse_mode", False)
    params = [
        ("Dataset",           json_name),
        ("Directory",         metadata.get("directory", "—")),
        ("Timestamp",         metadata.get("timestamp", "—")),
        ("Method",            metadata.get("divergence_method", "Jensen-Shannon + KL")),
        ("Total epochs",      str(metadata.get("total_epochs", "—"))),
        ("Baseline epochs",   (f"{metadata.get('baseline_epoch_count','—')}"
                               f"  ({metadata.get('baseline_percentage','?')}%)")),
        ("Attack epochs",     str(n_epochs)),
        ("Malicious traffic", f"{metadata.get('malicious_percentage','?')}%"),
        ("Epoch group size",  f"{GROUP_SIZE} epochs per point"),
        ("Metrics plotted",   str(len(METRICS))),
        ("Mode",              "SPARSE" if is_sparse else "FULL"),
    ]
    if is_sparse:
        params += [
            ("Attack probability", str(metadata.get("sparse_attack_prob", "—"))),
            ("Windows total",      str(metadata.get("total_windows", "—"))),
            ("Windows attacked",   str(metadata.get("attacked_window_count", "—"))),
            ("Windows clean",      str(metadata.get("clean_window_count", "—"))),
        ]

    ax.text(0.08, 0.855, "Run Parameters",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#333333")
    y = 0.825
    for k, v in params:
        ax.text(0.08, y, k, ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#666666")
        ax.text(0.30, y, v, ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#111111")
        y -= 0.043

    # Baseline summary (right column)
    ax.text(0.55, 0.855, "Baseline Distribution Summary",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#333333")

    baseline_rows = [
        ("total_flows",          "Total Flows"),
        ("unique_src_ips",       "Unique Src IPs"),
        ("unique_dst_ips",       "Unique Dst IPs"),
        ("unique_src_ports",     "Unique Src Ports"),
        ("unique_dst_ports",     "Unique Dst Ports"),
        ("src_ip_vocab_size",    "Src IP Vocab Size"),
        ("dst_ip_vocab_size",    "Dst IP Vocab Size"),
        ("src_port_vocab_size",  "Src Port Vocab Size"),
        ("dst_port_vocab_size",  "Dst Port Vocab Size"),
    ]

    y = 0.825
    for key, lbl in baseline_rows:
        val = baseline.get(key, "—")
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        ax.text(0.55, y, lbl, ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#666666")
        ax.text(0.80, y, val_str, ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#111111")
        y -= 0.043

    # Footer
    ax.plot([0.08, 0.92], [0.04, 0.04],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.5, 0.025,
            "Each subsequent page shows one metric. "
            "Points = average divergence delta over a group of 10 epochs.",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8, color="#999999", fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-metric page
# ---------------------------------------------------------------------------

def draw_metric_page(pdf: PdfPages, label: str, category: str,
                     x_ticks: List[int], x_labels: List[str],
                     y_vals: List[float],
                     malicious_pct, baseline_pct,
                     attacked_groups: set = None,
                     y_is_divergence: bool = True):
    """
    y_vals          – grouped mean of absolute divergence_vs_baseline values
    y_is_divergence – if True format y-axis as JS/KL units [0,1]; else integer Flow counts
    """
    color = CATEGORY_COLOR.get(category, "#333333")

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.84, bottom=0.13)

    # ── Reference line at 0 ────────────────────────────────────────────────────
    ax.axhline(0, color="#BBBBBB", linewidth=0.9, linestyle="--", zorder=1)

    ax.plot(x_ticks, y_vals,
            color=color, linewidth=1.0, alpha=0.35, zorder=2)

    above = [(x, y) for x, y in zip(x_ticks, y_vals) if y > 0]
    below = [(x, y) for x, y in zip(x_ticks, y_vals) if y <= 0]
    if above:
        ax.scatter(*zip(*above), color="#DC2626", s=48, zorder=5,
                   edgecolors="white", linewidths=0.6)
    if below:
        ax.scatter(*zip(*below), color="#2563EB", s=48, zorder=5,
                   edgecolors="white", linewidths=0.6)

    if y_is_divergence:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))
        ax.set_ylabel("Divergence vs baseline  (attacked epoch)", labelpad=8)
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylabel("Flow count  (attacked epoch)", labelpad=8)

    step = max(1, math.ceil(len(x_ticks) / 14))
    shown = x_ticks[::step]
    ax.set_xticks(shown)
    ax.set_xticklabels([x_labels[i] for i in shown],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(x_ticks) - 0.4)
    ax.set_xlabel(f"Epoch group  (each point = {GROUP_SIZE} epochs)", labelpad=6)

    if attacked_groups:
        for tick, tick_idx in zip(ax.get_xticklabels(), shown):
            if tick_idx in attacked_groups:
                tick.set_color("#DC2626")
                tick.set_fontweight("bold")
        for g in attacked_groups:
            ax.axvspan(g - 0.5, g + 0.5, color="#DC2626", alpha=0.07, zorder=0)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.93, label,
             ha="center", va="bottom",
             fontsize=15, fontweight="bold", color="#111111")
    fig.text(0.5, 0.895, category,
             ha="center", va="bottom",
             fontsize=9, color=color, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC2626",
               markersize=6, label="Higher divergence vs baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
               markersize=6, label="Lower / no divergence vs baseline"),
    ]
    if attacked_groups:
        handles.append(
            Patch(facecolor="#DC2626", alpha=0.15, label="Attacked window"))
    ax.legend(handles=handles, frameon=False, fontsize=8,
              loc="upper right", handletextpad=0.4)

    # ── Footer stats ──────────────────────────────────────────────────────────
    mean_v = float(np.mean(y_vals)) if y_vals else 0.0
    max_v  = float(np.max(y_vals))  if y_vals else 0.0
    min_v  = float(np.min(y_vals))  if y_vals else 0.0
    fmt = ".6f" if y_is_divergence else ",.0f"
    fig.text(0.5, 0.025,
             f"mean = {mean_v:{fmt}}     "
             f"max = {max_v:{fmt}}     "
             f"min = {min_v:{fmt}}     "
             f"malicious = {malicious_pct}%     "
             f"baseline epochs = {baseline_pct}%",
             ha="center", va="bottom",
             fontsize=7.5, color="#999999", fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_single_json(json_path: str, output_dir: str) -> bool:
    try:
        with open(json_path) as f:
            data: Dict[str, Any] = json.load(f)

        baseline: Dict       = data["baseline"]
        epochs:   List[Dict] = data["epochs"]
        metadata             = data.get("metadata", {})

        if not epochs:
            print(f"  [WARNING] No epoch data in {os.path.basename(json_path)}, skipping.")
            return False

        mal_pct   = metadata.get("malicious_percentage", "0")
        base_pct  = metadata.get("baseline_percentage",  "5")
        json_stem = os.path.splitext(os.path.basename(json_path))[0]

        pdf_path = os.path.join(output_dir, f"{json_stem}.pdf")

        epoch_indices = [e["epoch_index"] for e in epochs]
        x_labels = group_labels(epoch_indices, GROUP_SIZE)
        x_ticks  = list(range(len(x_labels)))

        # Sparse: map attacked windows → x-axis group indices
        is_sparse = metadata.get("sparse_mode", False)
        attacked_groups: set = set()
        if is_sparse:
            epoch_to_group = {ep: g
                              for g in range(len(x_labels))
                              for ep in epoch_indices[g*GROUP_SIZE:(g+1)*GROUP_SIZE]}
            for w in metadata.get("attacked_windows", []):
                for ep in w.get("epoch_indices", []):
                    grp = epoch_to_group.get(ep)
                    if grp is not None:
                        attacked_groups.add(grp)

        # Pre-compute grouped series for each metric
        grouped: Dict[str, List[float]] = {}

        for key, label, _ in METRICS:
            vals = []
            for ep in epochs:
                # Read absolute divergence of attacked epoch vs baseline
                v = ep.get("divergence_vs_baseline", {}).get(key, 0.0)
                vals.append(v)
            grouped[label] = group_avg(vals, GROUP_SIZE)

        with PdfPages(pdf_path) as pdf:
            draw_summary(pdf, metadata, baseline, len(epochs),
                         os.path.basename(json_path))

            for key, label, category in METRICS:
                y_is_div = (category in ("JS Divergence", "KL Divergence", "Combined"))
                draw_metric_page(
                    pdf             = pdf,
                    label           = label,
                    category        = category,
                    x_ticks         = x_ticks,
                    x_labels        = x_labels,
                    y_vals          = grouped[label],
                    malicious_pct   = mal_pct,
                    baseline_pct    = base_pct,
                    attacked_groups = attacked_groups if is_sparse else None,
                    y_is_divergence = y_is_div,
                )

            info = pdf.infodict()
            info["Title"]   = f"JS/KL Divergence — {json_stem}"
            info["Subject"] = f"Baseline {base_pct}% | Malicious {mal_pct}%"

        print(f"  [✓] {json_stem}.pdf")
        return True

    except Exception as e:
        import traceback
        print(f"  [ERROR] Failed to process {os.path.basename(json_path)}: {e}")
        traceback.print_exc()
        return False


def main():
    import glob

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

    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    if not json_files:
        print(f"[ERROR] No JSON files found in: {json_folder}")
        sys.exit(1)

    print("=" * 80)
    print("JS / KL DIVERGENCE PLOT GENERATOR — BATCH MODE")
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
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successful: {successful}/{len(json_files)}")
    if failed:
        print(f"Failed    : {failed}/{len(json_files)}")
    print(f"PDFs saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()