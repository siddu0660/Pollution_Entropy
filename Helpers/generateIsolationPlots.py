import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def analyze_json_results(json_path: str, output_dir: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metadata = data.get("metadata", {})
    epochs = data.get("epochs", [])
    
    if not epochs:
        print(f"Skipping {os.path.basename(json_path)}: No epoch data found.")
        return

    # Extract data arrays
    epoch_indices = []
    y_true = []
    y_pred = []
    anomaly_scores = []
    
    for ep in epochs:
        # Note: The original ML script didn't process baseline epochs in the output JSON, 
        # so this visualizes the test/attack pool.
        epoch_indices.append(ep.get("epoch_index", 0))
        
        # Ground truth: True if attacked, False if clean
        y_true.append(ep.get("attacked_ground_truth", False))
        
        # Prediction: True if anomaly detected (-1 in IsolationForest), False if normal
        y_pred.append(ep.get("ml_prediction_anomaly", False))
        
        # Raw anomaly score (negative means anomaly)
        anomaly_scores.append(ep.get("ml_anomaly_score", 0.0))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    anomaly_scores = np.array(anomaly_scores)
    epoch_indices = np.array(epoch_indices)

    # 1. Calculate Metrics
    print(f"\n" + "="*50)
    print(f"ANALYSIS FOR: {os.path.basename(json_path)}")
    print("="*50)
    
    # Only print classification report if there's a mix of clean and attacked data
    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["Clean", "Attacked"]))
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"True Positives (Attacks Caught): {tp}")
        print(f"False Positives (False Alarms):  {fp}")
        print(f"False Negatives (Missed Attacks):{fn}")
        print(f"True Negatives (Correct Clean):  {tn}")
        print(f"Detection Rate (TPR): {tpr:.2%}")
        print(f"False Alarm Rate (FPR): {fpr:.2%}")
    else:
        print("Note: Data only contains one class (either all attacked or all clean).")
        cm = confusion_matrix(y_true, y_pred)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Set up plotting style
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # PLOT 1: Anomaly Scores over Epochs
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    clean_mask = ~y_true
    attacked_mask = y_true
    
    plt.scatter(epoch_indices[clean_mask], anomaly_scores[clean_mask], 
                color='blue', label='Clean (Ground Truth)', alpha=0.6, marker='o')
    plt.scatter(epoch_indices[attacked_mask], anomaly_scores[attacked_mask], 
                color='red', label='Attacked (Ground Truth)', alpha=0.8, marker='x', s=60)
    
    # Decision boundary for Isolation Forest is 0
    plt.axhline(y=0, color='black', linestyle='--', label='Decision Boundary (y=0)')
    plt.fill_between([epoch_indices.min(), epoch_indices.max()], -0.5, 0, color='red', alpha=0.05, label='Anomaly Zone')
    
    plt.title(f'Isolation Forest Anomaly Scores Over Time\n({base_name})', fontsize=14)
    plt.xlabel('Epoch Index', fontsize=12)
    plt.ylabel('Anomaly Score (Negative = Anomaly)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plot1_path = os.path.join(output_dir, f"{base_name}_timeline.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Distribution of Scores
    # ---------------------------------------------------------
    if len(np.unique(y_true)) > 1:
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores[clean_mask], color='blue', label='Clean', kde=True, stat="density", alpha=0.4)
        sns.histplot(anomaly_scores[attacked_mask], color='red', label='Attacked', kde=True, stat="density", alpha=0.4)
        plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
        plt.title(f'Distribution of Anomaly Scores\n({base_name})', fontsize=14)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        plot2_path = os.path.join(output_dir, f"{base_name}_distribution.png")
        plt.savefig(plot2_path, dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Confusion Matrix
    # ---------------------------------------------------------
    if len(np.unique(y_true)) > 1:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred: Clean', 'Pred: Attacked'],
                    yticklabels=['Actual: Clean', 'Actual: Attacked'])
        plt.title(f'Confusion Matrix\n({base_name})', fontsize=14)
        plt.tight_layout()
        
        plot3_path = os.path.join(output_dir, f"{base_name}_confusion_matrix.png")
        plt.savefig(plot3_path, dpi=300)
        plt.close()

    print(f"Graphs saved to: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot Results from ML JSON files")
    parser.add_argument("directory", nargs="?", default=None, help="Directory containing the JSON files")
    args = parser.parse_args()

    if args.directory:
        json_dir = args.directory
    else:
        json_dir = input("\nEnter the directory path containing the JSON result files: ").strip()

    # Check if the user pointed to the root dataset dir or the json dir directly
    if not json_dir.endswith("json") and os.path.isdir(os.path.join(json_dir, "json")):
        json_dir = os.path.join(json_dir, "json")

    plot_output_dir = os.path.join(json_dir, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {plot_output_dir}")

    # Find all JSON files generated by the ML script
    json_files = glob.glob(os.path.join(json_dir, "*_ml*.json"))
    
    if not json_files:
        print(f"No JSON files matching '*_ml*.json' found in {json_dir}")
        return

    for json_file in sorted(json_files):
        try:
            analyze_json_results(json_file, plot_output_dir)
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")

if __name__ == "__main__":
    main()