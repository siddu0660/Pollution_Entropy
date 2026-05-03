import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report

# Apply styling
BG_COLOR = '#eef2f5'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": BG_COLOR, "figure.facecolor": BG_COLOR})


def _pick_sketch_example_epochs(epochs):
    """Pick one clean-like and one strongly anomalous epoch that carry sketch_preview payloads."""
    with_preview = [
        e
        for e in epochs
        if isinstance(e.get("sketch_preview"), dict)
        and e["sketch_preview"].get("master")
        and e["sketch_preview"].get("epoch")
    ]
    if not with_preview:
        return None, None
    clean = [e for e in with_preview if not e.get("attacked_ground_truth")]
    atk = [e for e in with_preview if e.get("attacked_ground_truth")]
    # Clean: least alarming score (closest to 0 from above); attack: worst (most negative) z-style score
    clean_pick = max(clean, key=lambda e: float(e.get("ml_anomaly_score", -1e9))) if clean else None
    atk_pick = min(atk, key=lambda e: float(e.get("ml_anomaly_score", 0.0))) if atk else None
    return clean_pick, atk_pick


def _build_cosine_sketch_figure(epoch, metadata, title, subtitle):
    """Line plot: master vs epoch slice, optional mean clean-baseline profile (same feature)."""
    sp = epoch["sketch_preview"]
    feat = sp.get("worst_feature", "?")
    master = np.asarray(sp["master"], dtype=float)
    obs = np.asarray(sp["epoch"], dtype=float)
    x = np.arange(master.size)

    fig = plt.figure(figsize=(9.0, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    ax.plot(x, master, label="Master $\\mathbf{m}$", color="#1f77b4", lw=2.0, alpha=0.9)
    ax.plot(x, obs, label="Epoch sketch $\\mathbf{v}$", color="#ff7f0e", lw=1.8, alpha=0.85)

    bmp = metadata.get("baseline_mean_sketch_preview") or {}
    if feat in bmp:
        baseline = np.asarray(bmp[feat], dtype=float)
        if baseline.size == master.size:
            ax.plot(
                x,
                baseline,
                label="Clean baseline (mean profile)",
                color="#2ca02c",
                lw=1.5,
                ls="--",
                alpha=0.9,
            )

    ax.set_xlabel("Index (first segment of flattened sketch)")
    ax.set_ylabel("Counter magnitude (normalised)")
    ax.set_title(title)
    ax.text(
        0.02,
        0.95,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    ax.text(
        0.98,
        0.95,
        f"feature: {feat}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.45),
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _export_cosine_sketch_assets(epoch, metadata, pdf, png_dir, base_name, kind_label):
    """Save one PNG and append the same figure to the open PdfPages handle."""
    if epoch is None:
        return
    sp = epoch["sketch_preview"]
    cos = sp.get("cosine", float("nan"))
    idx = epoch.get("epoch_index", "?")
    ename = epoch.get("epoch_name", "")
    title = f"{kind_label} (epoch {idx} {ename})".strip()
    if kind_label.lower().startswith("attack") or epoch.get("attacked_ground_truth"):
        subtitle = (
            f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}) \\approx {cos:.3f}$ on worst-z feature "
            "(lower — shape breaks baseline)"
        )
    else:
        subtitle = (
            f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}) \\approx {cos:.3f}$ on worst-z feature "
            "(typical clean drift)"
        )

    fig = _build_cosine_sketch_figure(epoch, metadata, title, subtitle)
    png_name = f"{base_name}_sketch_{kind_label.replace(' ', '_').lower()}.png"
    fig.savefig(os.path.join(png_dir, png_name), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

def create_text_page(pdf, title, text_content):
    """Creates a text-only page in the PDF for metrics and reports."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(BG_COLOR)
    plt.axis('off')
    
    plt.text(0.5, 0.95, title, fontsize=18, fontweight='bold', ha='center', va='top')
    plt.text(0.1, 0.85, text_content, fontsize=12, family='monospace', ha='left', va='top', wrap=True)
    
    pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

def analyze_to_pdf(json_path: str, output_dir: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metadata = data.get("metadata", {})
    epochs = data.get("epochs", [])
    
    if not epochs:
        print(f"Skipping {os.path.basename(json_path)}: No epoch data found.")
        return

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    
    is_cosine = metadata.get("method") == "Cosine_Similarity"
    
    # Extract data arrays
    epoch_indices, y_true, y_pred, anomaly_scores = [], [], [], []
    feature_z_scores = {'src_ip': [], 'dst_ip': [], 'src_port': [], 'dst_port': []} if is_cosine else None
    
    for ep in epochs:
        epoch_indices.append(ep.get("epoch_index", 0))
        y_true.append(ep.get("attacked_ground_truth", False))
        y_pred.append(ep.get("ml_prediction_anomaly", False))
        anomaly_scores.append(ep.get("ml_anomaly_score", 0.0))
        
        if is_cosine:
            z_scores = ep.get("z_scores", {})
            for feat in feature_z_scores.keys():
                feature_z_scores[feat].append(z_scores.get(feat, 0.0))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    anomaly_scores, epoch_indices = np.array(anomaly_scores), np.array(epoch_indices)
    clean_mask, attacked_mask = ~y_true, y_true

    print(f"\nProcessing {base_name} -> PDF...")

    with PdfPages(pdf_path) as pdf:
        # ---------------------------------------------------------
        # PAGE 1: Metrics and Classification Report
        # ---------------------------------------------------------
        report_text = f"Dataset: {metadata.get('dataset_name', 'Unknown')}\n"
        report_text += f"Analysis Method: {'Cosine Similarity' if is_cosine else 'Isolation Forest'}\n"
        report_text += f"Sparse Mode: {metadata.get('sparse_mode', False)}\n"
        report_text += "-" * 50 + "\n\n"
        
        cm = confusion_matrix(y_true, y_pred)
        if len(np.unique(y_true)) > 1:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            report_text += "CONFUSION MATRIX STATS:\n"
            report_text += f"True Positives (Attacks Caught):  {tp}\n"
            report_text += f"False Positives (False Alarms):   {fp}\n"
            report_text += f"False Negatives (Missed Attacks): {fn}\n"
            report_text += f"True Negatives (Correct Clean):   {tn}\n\n"
            
            report_text += f"Detection Rate (TPR):   {tpr:.2%}\n"
            report_text += f"False Alarm Rate (FPR): {fpr:.2%}\n\n"
            report_text += "-" * 50 + "\n\n"
            report_text += "CLASSIFICATION REPORT:\n"
            report_text += classification_report(y_true, y_pred, target_names=["Clean", "Attacked"])
        else:
            report_text += "Note: Data only contains one class (either all attacked or all clean).\n"
            report_text += f"Raw Confusion Matrix:\n{cm}"

        create_text_page(pdf, "Analysis Summary Report", report_text)

        # ---------------------------------------------------------
        # PAGE 2: Timeline Plot
        # ---------------------------------------------------------
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(epoch_indices[clean_mask], anomaly_scores[clean_mask], color='blue', label='Clean', alpha=0.6)
        plt.scatter(epoch_indices[attacked_mask], anomaly_scores[attacked_mask], color='red', label='Attacked', alpha=0.8, marker='x', s=60)
        
        threshold = -3.0 if is_cosine else 0.0
        plt.axhline(y=threshold, color='black', linestyle='--', label=f'Decision Boundary (y={threshold})')
        plt.fill_between([epoch_indices.min(), epoch_indices.max()], anomaly_scores.min() - 1, threshold, color='red', alpha=0.05, label='Anomaly Zone')
        
        title = "Cosine Similarity Worst Z-Score Over Time" if is_cosine else "Isolation Forest Anomaly Scores Over Time"
        plt.title(title, fontsize=14)
        plt.xlabel('Epoch Index', fontsize=12)
        plt.ylabel('Anomaly Score (Negative = Anomaly)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        # ---------------------------------------------------------
        # Cosine: master vs epoch sketch (+ clean mean baseline), PNGs + PDF pages
        # (requires sketch_preview in JSON from Entropy_Cosine.py)
        # ---------------------------------------------------------
        if is_cosine:
            clean_ep, atk_ep = _pick_sketch_example_epochs(epochs)
            if clean_ep is not None or atk_ep is not None:
                sketch_dir = os.path.join(output_dir, "sketch_pngs")
                os.makedirs(sketch_dir, exist_ok=True)
                if clean_ep is not None:
                    _export_cosine_sketch_assets(
                        clean_ep, metadata, pdf, sketch_dir, base_name, "Benign epoch vs master"
                    )
                if atk_ep is not None:
                    _export_cosine_sketch_assets(
                        atk_ep, metadata, pdf, sketch_dir, base_name, "Attack epoch vs master"
                    )
                print(f"  Sketch line-plot PNGs -> {sketch_dir}")

        # ---------------------------------------------------------
        # PAGE 3: Secondary Breakdown Plot
        # ---------------------------------------------------------
        if is_cosine:
            # 2x2 Feature Breakdown for Cosine
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
            fig.suptitle('Z-Scores per Feature Dimension', fontsize=16)
            features = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
            for i, feat in enumerate(features):
                ax = axes.flatten()[i]
                feat_z = np.array(feature_z_scores[feat])
                ax.scatter(epoch_indices[clean_mask], feat_z[clean_mask], color='blue', alpha=0.5, marker='.')
                ax.scatter(epoch_indices[attacked_mask], feat_z[attacked_mask], color='red', alpha=0.8, marker='x')
                ax.axhline(y=-3.0, color='black', linestyle=':', alpha=0.7)
                ax.set_title(f'Feature: {feat.upper()}')
            plt.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
        else:
            # Score Distribution for Isolation Forest
            if len(np.unique(y_true)) > 1:
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(anomaly_scores[clean_mask], color='blue', label='Clean', kde=True, stat="density", alpha=0.4)
                sns.histplot(anomaly_scores[attacked_mask], color='red', label='Attacked', kde=True, stat="density", alpha=0.4)
                plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
                plt.title('Distribution of Anomaly Scores', fontsize=14)
                plt.xlabel('Anomaly Score')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close(fig)

        # ---------------------------------------------------------
        # PAGE 4: Confusion Matrix Heatmap
        # ---------------------------------------------------------
        if len(np.unique(y_true)) > 1:
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if not is_cosine else 'Reds',
                        xticklabels=['Pred: Clean', 'Pred: Attacked'],
                        yticklabels=['Actual: Clean', 'Actual: Attacked'])
            plt.title('Confusion Matrix', fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export ML/Cosine JSON results to PDF reports")
    parser.add_argument("directory", nargs="?", default=None, help="Directory containing the JSON files")
    args = parser.parse_args()

    target_dir = args.directory if args.directory else input("\nEnter the directory path containing the JSON files: ").strip()

    if not target_dir.endswith("json") and os.path.isdir(os.path.join(target_dir, "json")):
        target_dir = os.path.join(target_dir, "json")

    pdf_output_dir = os.path.join(target_dir, "reports")
    os.makedirs(pdf_output_dir, exist_ok=True)
    print(f"\nPDF reports will be saved to: {pdf_output_dir}")

    json_files = glob.glob(os.path.join(target_dir, "*_ml*.json"))
    
    if not json_files:
        print(f"No valid JSON files found in {target_dir}")
        return

    for json_file in sorted(json_files):
        try:
            analyze_to_pdf(json_file, pdf_output_dir)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            
    print(f"\nAll PDFs generated successfully in {pdf_output_dir}!")

if __name__ == "__main__":
    main()