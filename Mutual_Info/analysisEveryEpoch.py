#!/usr/bin/env python3
"""
analysis.py
===========

Analyze MI detector JSON reports from a directory and generate:
1) Which MI pair triggers the most deviations
2) Accuracy / precision / recall vs malicious% (line plots per baseline%) at chosen z-score thresholds

Supports filename patterns such as:
- mi_<baseline>_<malicious>_<threshold>.json
- epochs_<baseline>_<malicious>_mi.json  (threshold falls back to metadata)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

MI_FEATURES = [
    "mi_src_dst_ip",
    "mi_src_ip_dst_port",
    "mi_src_ip_src_port",
    "mi_dst_ip_dst_port",
    "mi_src_dst_port",
    "mi_protocol_dst_port",
]

PAIR_NAMES = {
    "mi_src_dst_ip": "Src-IP x Dst-IP",
    "mi_src_ip_dst_port": "Src-IP x Dst-Port",
    "mi_src_ip_src_port": "Src-IP x Src-Port",
    "mi_dst_ip_dst_port": "Dst-IP x Dst-Port",
    "mi_src_dst_port": "Src-Port x Dst-Port",
    "mi_protocol_dst_port": "Protocol x Dst-Port",
}

FILENAME_PATTERNS = [
    # mi_5.0_1.0_2.5.json
    re.compile(
        r"^mi_(?P<baseline>\d+(?:\.\d+)?)_(?P<mal>\d+(?:\.\d+)?)_(?P<threshold>\d+(?:\.\d+)?)\.json$"
    ),
    # epochs_5.0_1.0_mi.json
    re.compile(
        r"^(?:.+_)?(?P<baseline>\d+(?:\.\d+)?)_(?P<mal>\d+(?:\.\d+)?)_mi\.json$"
    ),
]


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_name_fields(file_path: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"baseline": None, "malicious": None, "threshold": None}
    name = file_path.name
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(name)
        if not match:
            continue
        out["baseline"] = safe_float(match.groupdict().get("baseline"))
        out["malicious"] = safe_float(match.groupdict().get("mal"))
        out["threshold"] = safe_float(match.groupdict().get("threshold"))
        break
    return out


def looks_like_mi_report(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "metadata" not in data or "baseline" not in data or "epochs" not in data:
        return False
    epochs = data.get("epochs", [])
    if not isinstance(epochs, list):
        return False
    if not epochs:
        return True
    first = epochs[0]
    return isinstance(first, dict) and isinstance(first.get("z_scores", {}), dict)


def get_ground_truth(epoch: dict) -> Optional[bool]:
    if "attacked_ground_truth" in epoch:
        return bool(epoch.get("attacked_ground_truth"))
    if "attacked" in epoch:
        return bool(epoch.get("attacked"))
    return None


def iter_json_files(json_dir: Path) -> Iterable[Path]:
    for path in sorted(json_dir.rglob("*.json")):
        if path.is_file():
            yield path


def load_reports(json_dir: Path) -> List[dict]:
    loaded: List[dict] = []
    for path in iter_json_files(json_dir):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not looks_like_mi_report(data):
            continue

        meta = data.get("metadata", {})
        from_name = parse_name_fields(path)
        baseline_pct = safe_float(meta.get("baseline_percentage"))
        malicious_pct = safe_float(meta.get("malicious_percentage"))
        file_threshold = safe_float(meta.get("z_threshold"))
        if baseline_pct is None:
            baseline_pct = from_name["baseline"]
        if malicious_pct is None:
            malicious_pct = from_name["malicious"]
        if file_threshold is None:
            file_threshold = from_name["threshold"]

        loaded.append(
            {
                "path": str(path),
                "data": data,
                "baseline_pct": baseline_pct,
                "malicious_pct": malicious_pct,
                "file_threshold": file_threshold,
            }
        )
    return loaded


def predict_anomaly(epoch: dict, baseline_stats: dict, z_threshold: float) -> Tuple[bool, List[str]]:
    z_scores = epoch.get("z_scores", {}) or {}
    triggered_pairs: List[str] = []

    for feature in MI_FEATURES:
        std = safe_float(baseline_stats.get(f"{feature}_std"))
        if std is not None and std <= 0.0:
            # Ignore unstable features with near-zero std (e.g. protocol/dst-port in some runs)
            continue
        z = safe_float(z_scores.get(f"{feature}_z"))
        if z is not None and abs(z) >= z_threshold:
            triggered_pairs.append(feature)

    return bool(triggered_pairs), triggered_pairs


def update_confusion_counts(
    pred: bool,
    truth: Optional[bool],
    counts: Dict[str, int],
) -> None:
    if truth is None:
        return
    counts["epochs_with_truth"] += 1
    if pred and truth:
        counts["tp"] += 1
    elif pred and not truth:
        counts["fp"] += 1
    elif (not pred) and truth:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def compute_binary_metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    total = counts["epochs_with_truth"]
    if total == 0:
        return {"accuracy": None, "precision": None, "recall": None}
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def top_trigger_pair(triggered_counter: Dict[str, int]) -> Tuple[Optional[str], int]:
    if not triggered_counter:
        return None, 0
    return max(triggered_counter.items(), key=lambda x: x[1])


def evaluate_report(report: dict, z_threshold: float) -> dict:
    data = report["data"]
    epochs = data.get("epochs", [])
    baseline_stats = data.get("baseline", {})

    confusion_counts = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "epochs_with_truth": 0,
    }
    triggered_counter: Dict[str, int] = defaultdict(int)

    for epoch in epochs:
        pred, triggered_pairs = predict_anomaly(epoch, baseline_stats, z_threshold)
        for pair in triggered_pairs:
            triggered_counter[pair] += 1

        truth = get_ground_truth(epoch)
        update_confusion_counts(pred=pred, truth=truth, counts=confusion_counts)

    metrics = compute_binary_metrics(confusion_counts)
    top_pair, top_pair_count = top_trigger_pair(triggered_counter)

    return {
        "path": report["path"],
        "baseline_pct": report["baseline_pct"],
        "malicious_pct": report["malicious_pct"],
        "z_threshold_used": z_threshold,
        "epochs_count": len(epochs),
        "epochs_with_truth": confusion_counts["epochs_with_truth"],
        "tp": confusion_counts["tp"],
        "fp": confusion_counts["fp"],
        "tn": confusion_counts["tn"],
        "fn": confusion_counts["fn"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "top_trigger_pair": top_pair,
        "top_trigger_pair_name": PAIR_NAMES.get(top_pair, top_pair) if top_pair else None,
        "top_trigger_pair_count": top_pair_count,
        "trigger_counts": dict(sorted(triggered_counter.items(), key=lambda x: x[1], reverse=True)),
    }


def build_metric_matrix(
    results: List[dict], metric_key: str
) -> Tuple[List[float], List[float], Dict[str, Dict[str, Optional[float]]]]:
    baselines = sorted(
        {r["baseline_pct"] for r in results if r["baseline_pct"] is not None and r.get(metric_key) is not None}
    )
    maliciouses = sorted(
        {r["malicious_pct"] for r in results if r["malicious_pct"] is not None and r.get(metric_key) is not None}
    )
    matrix: Dict[str, Dict[str, Optional[float]]] = {}

    grouped: Dict[Tuple[float, float], List[float]] = defaultdict(list)
    for r in results:
        b = r["baseline_pct"]
        m = r["malicious_pct"]
        metric_val = r.get(metric_key)
        if b is None or m is None or metric_val is None:
            continue
        grouped[(b, m)].append(metric_val)

    for b in baselines:
        b_key = f"{b:g}"
        matrix[b_key] = {}
        for m in maliciouses:
            vals = grouped.get((b, m), [])
            matrix[b_key][f"{m:g}"] = (sum(vals) / len(vals)) if vals else None
    return baselines, maliciouses, matrix


def build_matrices_by_threshold(all_results: List[dict]) -> List[dict]:
    by_threshold: List[dict] = []
    thresholds = sorted({r["z_threshold_used"] for r in all_results})
    for zt in thresholds:
        subset = [r for r in all_results if r["z_threshold_used"] == zt]
        b_acc, m_acc, acc = build_metric_matrix(subset, "accuracy")
        b_pre, m_pre, pre = build_metric_matrix(subset, "precision")
        b_rec, m_rec, rec = build_metric_matrix(subset, "recall")
        by_threshold.append(
            {
                "z_threshold": zt,
                "accuracy_matrix": {
                    "rows_baseline_percent": b_acc,
                    "cols_malicious_percent": m_acc,
                    "values": acc,
                },
                "precision_matrix": {
                    "rows_baseline_percent": b_pre,
                    "cols_malicious_percent": m_pre,
                    "values": pre,
                },
                "recall_matrix": {
                    "rows_baseline_percent": b_rec,
                    "cols_malicious_percent": m_rec,
                    "values": rec,
                },
            }
        )
    return by_threshold


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "   n/a"
    return f"{value * 100:6.2f}%"


def print_matrix(
    title: str,
    baselines: List[float],
    maliciouses: List[float],
    matrix: Dict[str, Dict[str, Optional[float]]],
) -> None:
    if not baselines or not maliciouses:
        print(f"\n{title}: no labeled data available.")
        return

    print(f"\n{title} (rows=baseline%, cols=malicious%)")
    header = ["baseline\\mal"] + [f"{m:g}" for m in maliciouses]
    col_width = max(max(len(h) for h in header), 12)
    print("".join(h.rjust(col_width) for h in header))
    for b in baselines:
        b_key = f"{b:g}"
        row = [b_key.rjust(col_width)]
        for m in maliciouses:
            row.append(format_metric(matrix[b_key][f"{m:g}"]).rjust(col_width))
        print("".join(row))


def add_summary_page(pdf: "PdfPages", report: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    most = report["most_deviating_pair"]
    summary_lines = [
        "Every-Epoch MI Analysis Report",
        "",
        f"Input directory: {report['input_directory']}",
        f"Files processed: {report['files_processed']}",
        f"Z-score mode: {report.get('z_score_mode', 'fixed')}",
        f"Unique z-thresholds in analysis: {report.get('z_thresholds_used', [])}",
        (
            f"Most deviating pair: {most.get('pair_name') or 'n/a'} "
            f"(key={most.get('pair_key')}, triggers={most.get('trigger_count')})"
        ),
    ]
    y = 0.95
    for line in summary_lines:
        ax.text(0.02, y, line, fontsize=11, va="top")
        y -= 0.06

    trigger_counts = list(most.get("all_pair_trigger_counts", {}).items())[:6]
    if trigger_counts:
        table_data = [[PAIR_NAMES.get(k, k), str(v)] for k, v in trigger_counts]
        table = ax.table(
            cellText=table_data,
            colLabels=["Top trigger pair", "Count"],
            cellLoc="center",
            bbox=[0.02, 0.05, 0.6, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_trigger_chart_page(pdf: "PdfPages", report: dict) -> None:
    all_triggers = report["most_deviating_pair"].get("all_pair_trigger_counts", {})
    if not all_triggers:
        return

    keys = list(all_triggers.keys())
    vals = [all_triggers[k] for k in keys]
    names = [PAIR_NAMES.get(k, k) for k in keys]
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.barh(names, vals, color="#4C78A8")
    ax.set_title("Trigger Counts by MI Pair")
    ax.set_xlabel("Trigger count")
    ax.grid(axis="x", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_metric_line_chart(axis: "plt.Axes", title: str, payload: dict) -> None:
    baselines = payload["rows_baseline_percent"]
    maliciouses = payload["cols_malicious_percent"]
    values_map = payload["values"]
    if not baselines or not maliciouses:
        axis.axis("off")
        axis.set_title(f"{title}\n(no data)")
        return

    x = list(range(len(maliciouses)))
    xtick_labels = [f"{m:g}" for m in maliciouses]
    cmap = plt.get_cmap("tab10")
    for i, b in enumerate(baselines):
        b_key = f"{b:g}"
        row = values_map.get(b_key, {})
        ys: List[float] = []
        for m in maliciouses:
            val = row.get(f"{m:g}")
            ys.append(float(val) if val is not None else float("nan"))
        axis.plot(
            x,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=f"baseline {b:g}%",
            color=cmap(i % 10),
        )

    axis.set_title(title)
    axis.set_xlabel("Malicious %")
    axis.set_ylabel("Metric (0–1)")
    axis.set_xticks(x)
    axis.set_xticklabels(xtick_labels)
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.3)
    if len(baselines) <= 12:
        axis.legend(loc="best", fontsize=7, ncol=2 if len(baselines) > 6 else 1)


def save_individual_metric_plot(
    plot_dir: Path,
    metric_title: str,
    z_label: str,
    payload: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw_metric_line_chart(ax, metric_title, payload)
    ax.set_title(f"{metric_title} @ Z={z_label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    img_path = plot_dir / f"{metric_title}_{z_label}.png"
    fig.savefig(img_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def add_metric_graphs_page(pdf: "PdfPages", report: dict) -> None:
    add_metric_graphs_page_with_export(pdf=pdf, report=report, plot_dir=None)


def add_metric_graphs_page_with_export(
    pdf: "PdfPages", report: dict, plot_dir: Optional[Path]
) -> None:
    matrix_sets = report.get("matrices_by_z_threshold", [])
    if not matrix_sets:
        matrix_sets = [
            {
                "z_threshold": report.get("z_score_used"),
                "accuracy_matrix": report["accuracy_matrix"],
                "precision_matrix": report["precision_matrix"],
                "recall_matrix": report["recall_matrix"],
            }
        ]

    for matrix_set in matrix_sets:
        zt = matrix_set.get("z_threshold")
        z_label = f"{zt:g}" if isinstance(zt, (float, int)) else str(zt)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metric_payloads = [
            (axes[0], "Accuracy", matrix_set["accuracy_matrix"]),
            (axes[1], "Precision", matrix_set["precision_matrix"]),
            (axes[2], "Recall", matrix_set["recall_matrix"]),
        ]
        for axis, title, payload in metric_payloads:
            draw_metric_line_chart(axis, title, payload)
        fig.suptitle(f"MI metrics vs malicious % @ Z-threshold {z_label}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            for _, title, payload in metric_payloads:
                save_individual_metric_plot(plot_dir, title, z_label, payload)
        plt.close(fig)


def add_per_file_table_page(pdf: "PdfPages", report: dict) -> None:
    per_file = sorted(
        report["per_file_results"],
        key=lambda x: (
            x.get("baseline_pct") if x.get("baseline_pct") is not None else -1,
            x.get("malicious_pct") if x.get("malicious_pct") is not None else -1,
        ),
    )
    table_rows: List[List[str]] = []
    for row in per_file:
        table_rows.append(
            [
                f"{row.get('baseline_pct', 'n/a')}",
                f"{row.get('malicious_pct', 'n/a')}",
                f"{row.get('z_threshold_used', 'n/a')}",
                format_metric(row.get("accuracy")).strip(),
                format_metric(row.get("precision")).strip(),
                format_metric(row.get("recall")).strip(),
                str(row.get("epochs_with_truth", 0)),
                PAIR_NAMES.get(row.get("top_trigger_pair"), row.get("top_trigger_pair") or "n/a"),
            ]
        )

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows[:30],
        colLabels=[
            "Baseline %",
            "Malicious %",
            "Z",
            "Accuracy",
            "Precision",
            "Recall",
            "Epochs",
            "Top Trigger Pair",
        ],
        cellLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    ax.set_title("Per-file Metrics Table (first 30 rows)", fontsize=12, pad=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(report: dict, pdf_path: Path, plot_dir: Optional[Path] = None) -> None:
    if not HAS_MPL:
        print("\n[WARNING] matplotlib not installed; skipping PDF generation.")
        return

    with PdfPages(str(pdf_path)) as pdf:
        add_summary_page(pdf, report)
        add_trigger_chart_page(pdf, report)
        add_metric_graphs_page_with_export(pdf, report, plot_dir=plot_dir)
        add_per_file_table_page(pdf, report)


def build_report(json_dir: Path, z_score: Optional[float]) -> dict:
    loaded = load_reports(json_dir)
    if not loaded:
        raise ValueError(f"No MI-style JSON reports found in: {json_dir}")

    all_results: List[dict] = []
    global_trigger_counts: Dict[str, int] = defaultdict(int)

    for report in loaded:
        z_used = z_score
        if z_used is None:
            z_used = report["file_threshold"] if report["file_threshold"] is not None else 2.5

        result = evaluate_report(report, z_used)
        all_results.append(result)
        for pair, count in result["trigger_counts"].items():
            global_trigger_counts[pair] += count

    best_pair = None
    best_pair_count = 0
    if global_trigger_counts:
        best_pair, best_pair_count = max(global_trigger_counts.items(), key=lambda x: x[1])

    baselines_acc, maliciouses_acc, matrix_acc = build_metric_matrix(all_results, "accuracy")
    baselines_pre, maliciouses_pre, matrix_pre = build_metric_matrix(all_results, "precision")
    baselines_rec, maliciouses_rec, matrix_rec = build_metric_matrix(all_results, "recall")
    z_thresholds_used = sorted({r["z_threshold_used"] for r in all_results})
    matrices_by_threshold = build_matrices_by_threshold(all_results)

    return {
        "input_directory": str(json_dir),
        "z_score_used": z_score,
        "z_score_mode": "per-file" if z_score is None else "fixed",
        "z_thresholds_used": z_thresholds_used,
        "matrices_by_z_threshold": matrices_by_threshold,
        "files_processed": len(all_results),
        "most_deviating_pair": {
            "pair_key": best_pair,
            "pair_name": PAIR_NAMES.get(best_pair, best_pair) if best_pair else None,
            "trigger_count": best_pair_count,
            "all_pair_trigger_counts": dict(
                sorted(global_trigger_counts.items(), key=lambda x: x[1], reverse=True)
            ),
        },
        "accuracy_matrix": {
            "rows_baseline_percent": baselines_acc,
            "cols_malicious_percent": maliciouses_acc,
            "values": matrix_acc,
        },
        "precision_matrix": {
            "rows_baseline_percent": baselines_pre,
            "cols_malicious_percent": maliciouses_pre,
            "values": matrix_pre,
        },
        "recall_matrix": {
            "rows_baseline_percent": baselines_rec,
            "cols_malicious_percent": maliciouses_rec,
            "values": matrix_rec,
        },
        "per_file_results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MI JSON reports and generate report JSON + PDF."
    )
    parser.add_argument(
        "json_dir",
        help="Directory containing MI report JSON files.",
    )
    parser.add_argument(
        "--z-score",
        type=float,
        default=None,
        help=(
            "Fixed z-score threshold for all files. "
            "If omitted, each file's own metadata threshold is used."
        ),
    )
    parser.add_argument(
        "--output",
        default="analysis_report.json",
        help="Output JSON path for computed report (default: analysis_report.json).",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Output PDF path for charts/tables (default: same name as --output with .pdf).",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Folder to save metric line-plot PNGs (default: <output_dir>/every).",
    )
    parser.add_argument(
        "--heatmap-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    if not json_dir.exists() or not json_dir.is_dir():
        raise SystemExit(f"Invalid json_dir: {json_dir}")

    report = build_report(json_dir=json_dir, z_score=args.z_score)

    most = report["most_deviating_pair"]
    print(f"Files processed: {report['files_processed']}")
    print(f"Z-score mode: {report['z_score_mode']}")
    print(f"Z-thresholds used: {report['z_thresholds_used']}")
    print(
        "Most deviating pair: "
        f"{most['pair_name'] or 'n/a'} "
        f"(key={most['pair_key']}, triggers={most['trigger_count']})"
    )

    matrix = report["accuracy_matrix"]
    print_matrix(
        title="Accuracy Matrix",
        baselines=matrix["rows_baseline_percent"],
        maliciouses=matrix["cols_malicious_percent"],
        matrix=matrix["values"],
    )
    precision = report["precision_matrix"]
    print_matrix(
        title="Precision Matrix",
        baselines=precision["rows_baseline_percent"],
        maliciouses=precision["cols_malicious_percent"],
        matrix=precision["values"],
    )
    recall = report["recall_matrix"]
    print_matrix(
        title="Recall Matrix",
        baselines=recall["rows_baseline_percent"],
        maliciouses=recall["cols_malicious_percent"],
        matrix=recall["values"],
    )

    output_path = Path(args.output).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis report: {output_path}")

    pdf_path = Path(args.pdf).expanduser().resolve() if args.pdf else output_path.with_suffix(".pdf")
    default_plot_dir = output_path.parent / "every"
    plot_dir_arg = args.plot_dir or args.heatmap_dir
    plot_dir = Path(plot_dir_arg).expanduser().resolve() if plot_dir_arg else default_plot_dir
    write_pdf_report(report, pdf_path, plot_dir=plot_dir)
    if HAS_MPL:
        print(f"Saved PDF report: {pdf_path}")
        print(f"Saved metric plot images in: {plot_dir}")


if __name__ == "__main__":
    main()
