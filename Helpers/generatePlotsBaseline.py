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
    "Entropy":     "#2563EB",   # blue
    "Cardinality": "#DC2626",   # red
    "Uniformity":  "#16A34A",   # green
    "Flow":        "#D97706",   # amber
    "Count":       "#7C3AED",   # violet
}

GROUP_SIZE = 1

# Epoch is flagged as "attack detected" when its normalised deviation
# (epoch_val - baseline) / baseline exceeds this fraction.
DETECTION_THRESHOLD = 0.05  # 5% above baseline

METRICS = [
    ("src_ip_entropy_change",            "src_ip_entropy",            "Source IP Entropy",                    "Entropy"),
    ("dst_ip_entropy_change",            "dst_ip_entropy",            "Destination IP Entropy",               "Entropy"),
    ("src_port_entropy_change",          "src_port_entropy",          "Source Port Entropy",                  "Entropy"),
    ("dst_port_entropy_change",          "dst_port_entropy",          "Destination Port Entropy",             "Entropy"),
    ("src_dst_ip_entropy_change",        "src_dst_ip_entropy",        "Src+Dst IP Pair Entropy",              "Entropy"),
    ("src_dst_port_entropy_change",      "src_dst_port_entropy",      "Src+Dst Port Pair Entropy",            "Entropy"),
    ("src_ip_cardinality_change",        "src_ip_cardinality",        "Source IP Cardinality",                "Cardinality"),
    ("dst_ip_cardinality_change",        "dst_ip_cardinality",        "Destination IP Cardinality",           "Cardinality"),
    ("src_port_cardinality_change",      "src_port_cardinality",      "Source Port Cardinality",              "Cardinality"),
    ("dst_port_cardinality_change",      "dst_port_cardinality",      "Destination Port Cardinality",         "Cardinality"),
    ("src_dst_ip_cardinality_change",    "src_dst_ip_cardinality",    "Src+Dst IP Pair Cardinality",          "Cardinality"),
    ("src_dst_port_cardinality_change",  "src_dst_port_cardinality",  "Src+Dst Port Pair Cardinality",        "Cardinality"),
    ("src_ip_uniformity_change",         "src_ip_uniformity",         "Source IP Uniformity",                 "Uniformity"),
    ("dst_ip_uniformity_change",         "dst_ip_uniformity",         "Destination IP Uniformity",            "Uniformity"),
    ("src_dst_ratio_change",             "src_dst_ratio",             "Src / Dst IP Ratio",                   "Flow"),
    ("total_flows_change",               "total_flows",               "Total Flows",                          "Flow"),
    ("unique_src_ips_change",            "unique_src_ips",            "Unique Source IPs",                    "Count"),
    ("unique_dst_ips_change",            "unique_dst_ips",            "Unique Destination IPs",               "Count"),
    ("unique_src_ports_change",          "unique_src_ports",          "Unique Source Ports",                  "Count"),
    ("unique_dst_ports_change",          "unique_dst_ports",          "Unique Destination Ports",             "Count"),
]


def normalize(epoch_val: float, base: float) -> float:
    """(epoch − baseline) / baseline.  Returns 0 if baseline is zero."""
    if not base or abs(base) < 1e-12:
        return 0.0
    return (epoch_val - base) / base

def group_avg(vals: List[float], n: int) -> List[float]:
    return [float(np.mean(vals[i:i+n])) for i in range(0, len(vals), n)]

def group_labels(indices: List[int], n: int) -> List[str]:
    out = []
    for i in range(0, len(indices), n):
        chunk = indices[i:i+n]
        out.append(f"E{chunk[0]+1}\u2013{chunk[-1]+1}")
    return out


def compute_confusion(
    per_epoch_deviations: Dict[str, List[float]],
    attacked_flags: List[bool],
    threshold: float = DETECTION_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """Per-metric confusion matrix against ground-truth attacked flags.

    Returns a dict  label -> {TP, FP, TN, FN, precision, recall, F1}.
    An epoch is predicted "attacked" when its normalised deviation > threshold.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for label, deviations in per_epoch_deviations.items():
        tp = fp = tn = fn = 0
        for dev, attacked in zip(deviations, attacked_flags):
            detected = dev > threshold
            if detected and attacked:
                tp += 1
            elif detected and not attacked:
                fp += 1
            elif not detected and not attacked:
                tn += 1
            else:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[label] = {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1,
        }
    return results


def draw_summary(pdf: PdfPages, metadata: dict, baseline: dict,
                 n_epochs: int, json_name: str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")

    # ── Title block ───────────────────────────────────────────────────────────
    ax.text(0.5, 0.95, "Entropy Stream Analysis",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="#111111")
    ax.text(0.5, 0.905, "Deviation from Baseline — Per-Metric Report",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=12, color="#666666")

    # thin rule under title
    ax.plot([0.08, 0.92], [0.885, 0.885],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)

    # ── Run parameters (left column) ─────────────────────────────────────────
    is_sparse = metadata.get("sparse_mode", False)

    params = [
        ("Dataset",          json_name),
        ("Directory",        metadata.get("directory", "—")),
        ("Timestamp",        metadata.get("timestamp", "—")),
        ("Total epochs",     str(metadata.get("total_epochs", "—"))),
        ("Baseline epochs",  (f"{metadata.get('baseline_epoch_count','—')}"
                              f"  ({metadata.get('baseline_percentage','?')}%)")),
        ("Attack epochs",    str(n_epochs)),
        ("Malicious traffic",f"{metadata.get('malicious_percentage','?')}%"),
        ("Epoch group size", "1 (every epoch plotted)"),
        ("Metrics plotted",  str(len(METRICS))),
        ("Mode",             "SPARSE" if is_sparse else "FULL"),
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
        ax.text(0.08, y, k,
                ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#666666")
        ax.text(0.28, y, v,
                ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#111111")
        y -= 0.043

    # ── Baseline values (right column) ───────────────────────────────────────
    ax.text(0.55, 0.855, "Baseline Values",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#333333")

    baseline_rows = [
        ("src_ip_entropy",           "Src IP Entropy"),
        ("dst_ip_entropy",           "Dst IP Entropy"),
        ("src_port_entropy",         "Src Port Entropy"),
        ("dst_port_entropy",         "Dst Port Entropy"),
        ("src_dst_ip_entropy",       "Src+Dst IP Entropy"),
        ("src_dst_port_entropy",     "Src+Dst Port Entropy"),
        ("src_ip_cardinality",       "Src IP Cardinality"),
        ("dst_ip_cardinality",       "Dst IP Cardinality"),
        ("src_port_cardinality",     "Src Port Cardinality"),
        ("dst_port_cardinality",     "Dst Port Cardinality"),
        ("src_dst_ip_cardinality",   "Src+Dst IP Cardinality"),
        ("src_dst_port_cardinality", "Src+Dst Port Cardinality"),
        ("src_ip_uniformity",        "Src IP Uniformity"),
        ("dst_ip_uniformity",        "Dst IP Uniformity"),
        ("src_dst_ratio",            "Src/Dst Ratio"),
        ("total_flows",              "Total Flows"),
    ]

    y = 0.825
    for key, lbl in baseline_rows:
        val = baseline.get(key, "—")
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        ax.text(0.55, y, lbl,
                ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#666666")
        ax.text(0.78, y, val_str,
                ha="left", va="top", transform=ax.transAxes,
                fontsize=9, color="#111111")
        y -= 0.043

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.plot([0.08, 0.92], [0.04, 0.04],
            color="#DDDDDD", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.5, 0.025,
            "Each subsequent page shows one metric. "
            "Each point = one epoch's normalised deviation from the baseline.",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8, color="#999999", fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ── Metric pages ─────────────────────────────────────────────────────────────

def draw_metric_page(pdf: PdfPages, label: str, category: str,
                     x_ticks: List[int], x_labels: List[str],
                     y_vals: List[float], baseline_val: float,
                     malicious_pct, baseline_pct,
                     attacked_groups: set = None):

    color = CATEGORY_COLOR.get(category, "#333333")

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.84, bottom=0.15)

    # ── Zero baseline reference ───────────────────────────────────────────────
    ax.axhline(0, color="#BBBBBB", linewidth=0.9, linestyle="--", zorder=1)

    # ── Connecting line (subtle, same colour, thin) ───────────────────────────
    ax.plot(x_ticks, y_vals,
            color=color, linewidth=1.0, alpha=0.35, zorder=2)

    # ── Scatter points ────────────────────────────────────────────────────────
    ax.scatter(x_ticks, y_vals,
               color=color, s=48, zorder=4, edgecolors="white",
               linewidths=0.6)

    # Colour-code above / below zero
    above = [(x, y) for x, y in zip(x_ticks, y_vals) if y > 0]
    below = [(x, y) for x, y in zip(x_ticks, y_vals) if y <= 0]
    if above:
        ax.scatter(*zip(*above), color="#DC2626", s=48, zorder=5,
                   edgecolors="white", linewidths=0.6)
    if below:
        ax.scatter(*zip(*below), color="#2563EB", s=48, zorder=5,
                   edgecolors="white", linewidths=0.6)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:+.3f}"))
    ax.set_ylabel("Normalized deviation  (epoch − base) / base", labelpad=8)
    ax.set_xlabel("Epoch", labelpad=8)

    # ── X-axis ticks ──────────────────────────────────────────────────────────
    step = max(1, math.ceil(len(x_ticks) / 14))
    shown = x_ticks[::step]
    ax.set_xticks(shown)
    ax.set_xticklabels([x_labels[i] for i in shown],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(x_ticks) - 0.4)

    # Colour attacked-window tick labels red
    if attacked_groups:
        for tick, tick_idx in zip(ax.get_xticklabels(), shown):
            if tick_idx in attacked_groups:
                tick.set_color("#DC2626")
                tick.set_fontweight("bold")

    # Shade attacked-window regions
    if attacked_groups:
        for g in attacked_groups:
            ax.axvspan(g - 0.5, g + 0.5,
                       color="#DC2626", alpha=0.07, zorder=0)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.93, label,
             ha="center", va="bottom",
             fontsize=15, fontweight="bold", color="#111111")
    fig.text(0.5, 0.895, category,
             ha="center", va="bottom",
             fontsize=9, color=color, fontweight="bold")

    # ── Minimal legend (top-right, no box) ───────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC2626",
               markersize=6, label="Above baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
               markersize=6, label="Below baseline"),
    ]
    if attacked_groups:
        handles.append(
            Patch(facecolor="#DC2626", alpha=0.15, label="Attacked window")
        )
    ax.legend(handles=handles, frameon=False, fontsize=8,
              loc="upper right", handletextpad=0.4)

    # ── Footer stats ──────────────────────────────────────────────────────────
    mean_d = float(np.mean(y_vals)) if y_vals else 0.0
    max_d  = float(np.max(np.abs(y_vals))) if y_vals else 0.0
    fig.text(0.5, 0.025,
             f"baseline = {baseline_val:.4g}     "
             f"mean norm = {mean_d:+.4f}     "
             f"max |norm| = {max_d:.4f}     "
             f"malicious = {malicious_pct}%     "
             f"baseline epochs = {baseline_pct}%",
             ha="center", va="bottom",
             fontsize=7.5, color="#999999", fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ── Confusion matrix page ─────────────────────────────────────────────────────

def draw_confusion_page(
    pdf: PdfPages,
    cm: Dict[str, Dict[str, Any]],
    malicious_pct,
    baseline_pct,
    threshold: float,
    n_attacked: int,
    n_clean: int,
) -> None:
    """Draw a table page summarising per-metric confusion matrices."""
    labels = list(cm.keys())
    cols   = ["TP", "FP", "TN", "FN", "Precision", "Recall", "F1"]

    table_data = []
    for lbl in labels:
        r = cm[lbl]
        table_data.append([
            r["TP"], r["FP"], r["TN"], r["FN"],
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['F1']:.3f}",
        ])

    n_rows = len(labels)
    fig_h  = max(6, 0.38 * n_rows + 3.0)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    fig.text(
        0.5, 0.97,
        f"Confusion Matrix — threshold {threshold*100:.0f}% above baseline",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#111111",
    )
    fig.text(
        0.5, 0.93,
        f"Malicious {malicious_pct}%  |  Baseline {baseline_pct}%  |  "
        f"Attacked epochs: {n_attacked}  |  Clean epochs: {n_clean}",
        ha="center", va="top", fontsize=9, color="#666666",
    )

    tbl = ax.table(
        cellText  = table_data,
        rowLabels = labels,
        colLabels = cols,
        cellLoc   = "center",
        rowLoc    = "right",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.55)

    # Header row styling
    for j, col in enumerate(cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")

    # Row label styling + FP highlight
    for i, lbl in enumerate(labels):
        # row label
        rc = tbl[i + 1, -1]
        rc.set_facecolor("#F0F4FA")
        rc.set_text_props(fontweight="bold", color="#1E3A5F")

        fp_val = cm[lbl]["FP"]
        for j in range(len(cols)):
            cell = tbl[i + 1, j]
            if j == 1:  # FP column
                if fp_val == 0:
                    cell.set_facecolor("#D1FAE5")   # green — no FPs
                elif fp_val <= 3:
                    cell.set_facecolor("#FEF3C7")   # amber
                else:
                    cell.set_facecolor("#FEE2E2")   # red
            elif i % 2 == 0:
                cell.set_facecolor("#F8FAFC")

    # Summary bar: total FPs across all metrics
    total_fp = sum(cm[l]["FP"] for l in labels)
    fig.text(
        0.5, 0.02,
        f"Total false positives across all metrics: {total_fp}   "
        f"(green = 0 FP, amber = 1-3 FP, red = >3 FP)",
        ha="center", va="bottom", fontsize=8, color="#555555", fontstyle="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.91])
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_single_json(json_path: str, output_dir: str) -> bool:
    try:
        with open(json_path) as f:
            data: Dict[str, Any] = json.load(f)

        baseline: Dict[str, float] = data["baseline"]
        epochs:   List[Dict]       = data["epochs"]
        metadata                   = data.get("metadata", {})

        if not epochs:
            print(f"  [WARNING] No epoch data in {os.path.basename(json_path)}, skipping.")
            return False

        mal_pct  = metadata.get("malicious_percentage", "0")
        base_pct = metadata.get("baseline_percentage",  "5")
        json_stem = os.path.splitext(os.path.basename(json_path))[0]

        pdf_name = f"{json_stem}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        epoch_indices = [e["epoch_index"] for e in epochs]
        x_labels = group_labels(epoch_indices, GROUP_SIZE)
        x_ticks  = list(range(len(x_labels)))

        # ── Sparse: map attacked window_index → group index on x-axis ─────────
        is_sparse = metadata.get("sparse_mode", False)
        attacked_groups: set = set()
        if is_sparse:
            # epoch_indices are the post-baseline epoch indices in order.
            # group_labels splits them into chunks of GROUP_SIZE, so group i
            # covers epoch_indices[i*GROUP_SIZE : (i+1)*GROUP_SIZE].
            # The JSON stores attacked_windows with their epoch_indices lists;
            # we find which group each attacked epoch falls into.
            epoch_to_group = {ep: g for g, ep in
                              ((g, ep)
                               for g in range(len(x_labels))
                               for ep in epoch_indices[g*GROUP_SIZE:(g+1)*GROUP_SIZE])}
            for w in metadata.get("attacked_windows", []):
                for ep in w.get("epoch_indices", []):
                    grp = epoch_to_group.get(ep)
                    if grp is not None:
                        attacked_groups.add(grp)

        # ── Per-epoch deviations & attacked flags (for confusion matrix) ──────
        attacked_flags = [bool(e.get("attacked", True)) for e in epochs]
        per_epoch_dev: Dict[str, List[float]] = {}
        for change_key, baseline_key, label, _ in METRICS:
            bval = baseline.get(baseline_key, 0.0)
            devs = [
                normalize(ep.get("malicious_metrics", {}).get(baseline_key, bval), bval)
                for ep in epochs
            ]
            per_epoch_dev[label] = devs

        cm = compute_confusion(per_epoch_dev, attacked_flags, DETECTION_THRESHOLD)
        n_attacked = sum(attacked_flags)
        n_clean    = len(attacked_flags) - n_attacked

        grouped: Dict[str, List[float]] = {}
        for change_key, baseline_key, label, _ in METRICS:
            per_ep = []
            bval = baseline.get(baseline_key, 0.0)
            for ep in epochs:
                # Use the malicious_metrics value directly for normalization
                epoch_val = ep.get("malicious_metrics", {}).get(baseline_key, bval)
                per_ep.append(normalize(epoch_val, bval))
            grouped[label] = group_avg(per_ep, GROUP_SIZE)

        with PdfPages(pdf_path) as pdf:
            draw_summary(pdf, metadata, baseline, len(epochs),
                         os.path.basename(json_path))

            draw_confusion_page(
                pdf           = pdf,
                cm            = cm,
                malicious_pct = mal_pct,
                baseline_pct  = base_pct,
                threshold     = DETECTION_THRESHOLD,
                n_attacked    = n_attacked,
                n_clean       = n_clean,
            )

            for change_key, baseline_key, label, category in METRICS:
                draw_metric_page(
                    pdf          = pdf,
                    label        = label,
                    category     = category,
                    x_ticks      = x_ticks,
                    x_labels     = x_labels,
                    y_vals       = grouped[label],
                    baseline_val = baseline.get(baseline_key, 0.0),
                    malicious_pct = mal_pct,
                    baseline_pct  = base_pct,
                    attacked_groups = attacked_groups if is_sparse else None,
                )

            info = pdf.infodict()
            info["Title"]   = f"Entropy Stream — {json_stem}"
            info["Subject"] = f"Baseline {base_pct}% | Malicious {mal_pct}%"

        print(f"  [✓] {pdf_name}  (FP total: {sum(cm[l]['FP'] for l in cm)})")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {os.path.basename(json_path)}: {str(e)}")
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

    parent_dir = os.path.dirname(os.path.abspath(json_folder))
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in: {json_folder}")
        sys.exit(1)

    print("="*80)
    print("ENTROPY BASELINE PLOT GENERATOR - BATCH MODE")
    print("="*80)
    print(f"\nJSON folder    : {json_folder}")
    print(f"Results folder : {results_dir}")
    print(f"JSON files     : {len(json_files)}")
    print("-"*80)

    successful = 0
    failed = 0
    
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {os.path.basename(json_path)}")
        if process_single_json(json_path, results_dir):
            successful += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successful: {successful}/{len(json_files)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(json_files)}")
    print(f"PDFs saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()