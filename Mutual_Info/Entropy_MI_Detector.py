"""
Entropy_MI_Detector.py — Mutual Information-based Attack Detector
===================================================================

Key Insight:
-----------
Mutual Information I(X;Y) measures the dependency between two features.
Attack traffic creates structured patterns with higher MI between features:
  - DDoS:      High MI(src_ips, dst_ip) - many sources → one target
  - Port Scan: High MI(src_ip, dst_ports) - one source → many ports
  - Normal:    Low MI - features are relatively independent

Approach:
---------
1. Baseline: Calculate MI between all feature pairs from clean benign epochs
2. Detection: Calculate MI on mixed traffic, compare against baseline using Z-scores
3. Flag: Epochs where MI deviates significantly from baseline patterns

Usage:
------
  python Entropy_MI_Detector.py <directory> [--sparse]
    (baseline/malicious/z-threshold vectors are hardcoded)
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import zlib
import math
import os
import random
import glob
import json
import statistics
from datetime import datetime


def format_numeric_token(value: float) -> str:
    """Stable numeric token for filenames (e.g., 5.0, 2.5, 2.25)."""
    return str(float(value))


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str


def parse_flows(flow_file: str) -> List[Flow]:
    flows = []
    with open(flow_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) != 36:
                continue
            flows.append(Flow(
                line[0:12], line[12:24], line[24:29],
                line[29:34], line[34:36]
            ))
    return flows


# ---------------------------------------------------------------------------
# Count-Min Sketch for probability distributions
# ---------------------------------------------------------------------------
class CMS:
    def __init__(self, width=512, depth=4):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]
        self.total = 0

    def _h(self, item: str, row: int) -> int:
        return zlib.crc32(f"{item}_{row}".encode()) % self.w

    def add(self, item: str):
        for r in range(self.d):
            self.table[r][self._h(item, r)] += 1
        self.total += 1

    def est(self, item: str) -> int:
        return min(self.table[r][self._h(item, r)] for r in range(self.d))

    def prob(self, item: str) -> float:
        """Probability estimate for single item"""
        if self.total == 0:
            return 0.0
        return self.est(item) / self.total


# ---------------------------------------------------------------------------
# Mutual Information Calculator
# ---------------------------------------------------------------------------
def calculate_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy H(X) from frequency dict"""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def calculate_joint_entropy(joint_counts: Dict[Tuple[str, str], int]) -> float:
    """Joint entropy H(X,Y) from joint frequency dict"""
    total = sum(joint_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in joint_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def calculate_mutual_information(flows: List[Flow],
                                 feature_x: str,
                                 feature_y: str) -> float:
    """
    Calculate MI(X;Y) = H(X) + H(Y) - H(X,Y)

    Features: 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'
    """
    if not flows:
        return 0.0

    # Extract features
    def get_feature(flow: Flow, feat: str) -> str:
        return getattr(flow, feat)

    # Count marginal distributions
    counts_x = Counter(get_feature(f, feature_x) for f in flows)
    counts_y = Counter(get_feature(f, feature_y) for f in flows)

    # Count joint distribution
    joint_counts = Counter(
        (get_feature(f, feature_x), get_feature(f, feature_y))
        for f in flows
    )

    # Calculate entropies
    h_x = calculate_entropy(counts_x)
    h_y = calculate_entropy(counts_y)
    h_xy = calculate_joint_entropy(joint_counts)

    # MI = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy

    return max(0.0, mi)  # MI is always non-negative


# ---------------------------------------------------------------------------
# Feature pairs to track
# ---------------------------------------------------------------------------
MI_FEATURE_PAIRS = [
    ('src_ip', 'dst_ip'),       # Source-destination coordination
    ('src_ip', 'dst_port'),     # Source targeting specific services
    ('src_ip', 'src_port'),     # Source port patterns
    ('dst_ip', 'dst_port'),     # Destination service patterns
    ('src_port', 'dst_port'),   # Port-to-port patterns
    ('protocol', 'dst_port'),   # Protocol-service correlation
]


@dataclass
class MIMetrics:
    """Mutual Information metrics for an epoch"""
    total_flows: int
    unique_src_ips: int
    unique_dst_ips: int
    unique_src_ports: int
    unique_dst_ports: int

    # MI values for each feature pair
    mi_src_dst_ip: float = 0.0
    mi_src_ip_dst_port: float = 0.0
    mi_src_ip_src_port: float = 0.0
    mi_dst_ip_dst_port: float = 0.0
    mi_src_dst_port: float = 0.0
    mi_protocol_dst_port: float = 0.0

    # Entropy values (for reference)
    h_src_ip: float = 0.0
    h_dst_ip: float = 0.0
    h_src_port: float = 0.0
    h_dst_port: float = 0.0


def calculate_mi_metrics(flows: List[Flow]) -> MIMetrics:
    """Calculate all MI metrics for a set of flows"""
    if not flows:
        return MIMetrics(0, 0, 0, 0, 0)

    metrics = MIMetrics(
        total_flows=len(flows),
        unique_src_ips=len(set(f.src_ip for f in flows)),
        unique_dst_ips=len(set(f.dst_ip for f in flows)),
        unique_src_ports=len(set(f.src_port for f in flows)),
        unique_dst_ports=len(set(f.dst_port for f in flows)),
    )

    # Calculate MI for each feature pair
    metrics.mi_src_dst_ip = calculate_mutual_information(flows, 'src_ip', 'dst_ip')
    metrics.mi_src_ip_dst_port = calculate_mutual_information(flows, 'src_ip', 'dst_port')
    metrics.mi_src_ip_src_port = calculate_mutual_information(flows, 'src_ip', 'src_port')
    metrics.mi_dst_ip_dst_port = calculate_mutual_information(flows, 'dst_ip', 'dst_port')
    metrics.mi_src_dst_port = calculate_mutual_information(flows, 'src_port', 'dst_port')
    metrics.mi_protocol_dst_port = calculate_mutual_information(flows, 'protocol', 'dst_port')

    # Calculate individual entropies
    metrics.h_src_ip = calculate_entropy(Counter(f.src_ip for f in flows))
    metrics.h_dst_ip = calculate_entropy(Counter(f.dst_ip for f in flows))
    metrics.h_src_port = calculate_entropy(Counter(f.src_port for f in flows))
    metrics.h_dst_port = calculate_entropy(Counter(f.dst_port for f in flows))

    return metrics


def metrics_to_dict(m: MIMetrics) -> dict:
    return {
        "total_flows": m.total_flows,
        "unique_src_ips": m.unique_src_ips,
        "unique_dst_ips": m.unique_dst_ips,
        "unique_src_ports": m.unique_src_ports,
        "unique_dst_ports": m.unique_dst_ports,
        "mi_src_dst_ip": round(m.mi_src_dst_ip, 6),
        "mi_src_ip_dst_port": round(m.mi_src_ip_dst_port, 6),
        "mi_src_ip_src_port": round(m.mi_src_ip_src_port, 6),
        "mi_dst_ip_dst_port": round(m.mi_dst_ip_dst_port, 6),
        "mi_src_dst_port": round(m.mi_src_dst_port, 6),
        "mi_protocol_dst_port": round(m.mi_protocol_dst_port, 6),
        "h_src_ip": round(m.h_src_ip, 6),
        "h_dst_ip": round(m.h_dst_ip, 6),
        "h_src_port": round(m.h_src_port, 6),
        "h_dst_port": round(m.h_dst_port, 6),
    }


# ---------------------------------------------------------------------------
# Baseline calculation
# ---------------------------------------------------------------------------
MI_FIELDS = [
    "mi_src_dst_ip", "mi_src_ip_dst_port", "mi_src_ip_src_port",
    "mi_dst_ip_dst_port", "mi_src_dst_port", "mi_protocol_dst_port",
    "h_src_ip", "h_dst_ip", "h_src_port", "h_dst_port",
    "total_flows", "unique_src_ips", "unique_dst_ips"
]


def calculate_baseline_stats(baseline_metrics_list: List[MIMetrics]) -> Dict[str, float]:
    """Calculate mean and std for all MI metrics from baseline epochs"""
    if not baseline_metrics_list:
        return {}

    baseline = {}
    for field in MI_FIELDS:
        vals = [getattr(m, field) for m in baseline_metrics_list]
        baseline[f"{field}_mean"] = statistics.mean(vals)
        baseline[f"{field}_std"] = statistics.stdev(vals) if len(vals) > 1 else 1e-9

    return baseline


def calculate_z_scores(baseline: Dict[str, float], current: MIMetrics) -> Dict[str, float]:
    """Calculate Z-scores for current metrics vs baseline"""
    z_scores = {}
    for field in MI_FIELDS:
        mean = baseline.get(f"{field}_mean", 0.0)
        std = baseline.get(f"{field}_std", 1e-9) or 1e-9
        current_val = getattr(current, field)
        z_scores[f"{field}_z"] = (current_val - mean) / std

    return z_scores


def detect_anomaly(z_scores: Dict[str, float],
                   threshold: float = 2.5) -> Tuple[bool, str, float]:
    """
    Detect if epoch is anomalous based on MI Z-scores

    Returns: (is_anomaly, reason, max_z_score)
    """
    # Check MI-specific features (most important for attack detection)
    mi_features = [
        "mi_src_dst_ip_z",
        "mi_src_ip_dst_port_z",
        "mi_dst_ip_dst_port_z",
    ]

    max_z = 0.0
    max_feature = ""

    for feat in mi_features:
        z = abs(z_scores.get(feat, 0.0))
        if z > max_z:
            max_z = z
            max_feature = feat

    # Also check entropy deviations
    entropy_features = ["h_src_ip_z", "h_dst_ip_z"]
    for feat in entropy_features:
        z = abs(z_scores.get(feat, 0.0))
        if z > max_z:
            max_z = z
            max_feature = feat

    if max_z > threshold:
        return True, f"threshold_breach: {max_feature}={z_scores.get(max_feature, 0):.2f}", max_z

    return False, "clean", max_z


# ---------------------------------------------------------------------------
# Malicious flow generation
# ---------------------------------------------------------------------------
def generate_malicious_flows(num_flows: int, base_flows: List[Flow]) -> List[Flow]:
    flows = []
    for _ in range(num_flows):
        src_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        dst_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        src_port = f"{random.randint(0,65535):05d}"
        dst_port_val = random.choice([80, 443, 808, 22, 53, 330, 543, random.randint(0,65535)])
        dst_port = f"{dst_port_val % 65535:05d}"
        protocol = f"{random.choice([6,17]):02d}"
        flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    return flows


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
def run_single_configuration(directory: str, baseline_pct: float, malicious_pct: float,
                             dataset_name: str, output_dir: str,
                             z_threshold: float = 2.5) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*'))
                            if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_pct / 100))

    print(f"\n  [MI DETECTOR] Baseline={baseline_pct}%, Malicious={malicious_pct}%")
    print(f"  Total epochs: {total_epochs}, Baseline: {baseline_count}")

    baseline_indices = list(range(baseline_count))
    attack_indices = list(range(baseline_count, total_epochs))

    # Phase 1: Build baseline from CLEAN benign flows
    print(f"  Phase 1: Building MI baseline from clean epochs...")
    baseline_metrics_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_metrics_list.append(calculate_mi_metrics(flows))

    baseline_stats = calculate_baseline_stats(baseline_metrics_list)
    print(f"  Baseline established from {len(baseline_metrics_list)} clean epochs")

    # Phase 2: Test on attack epochs
    print(f"  Phase 2: Analyzing attack epochs...")
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "Mutual Information Detector",
            "directory": directory,
            "total_epochs": total_epochs,
            "baseline_epoch_count": baseline_count,
            "baseline_percentage": baseline_pct,
            "malicious_percentage": malicious_pct,
            "z_threshold": z_threshold,
            "sparse_mode": False,
        },
        "baseline": {k: round(v, 6) if isinstance(v, float) else v
                    for k, v in baseline_stats.items()},
        "epochs": []
    }

    detected_count = 0
    for idx in attack_indices:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]

        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        # Inject malicious traffic
        num_malicious = int(len(clean_flows) * (malicious_pct / 100))
        malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
        combined_flows = clean_flows + malicious_flows

        # Calculate MI metrics
        observed_metrics = calculate_mi_metrics(combined_flows)
        z_scores = calculate_z_scores(baseline_stats, observed_metrics)
        is_anomaly, reason, max_z = detect_anomaly(z_scores, z_threshold)

        if is_anomaly:
            detected_count += 1

        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked": True,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "mi_metrics": metrics_to_dict(observed_metrics),
            "z_scores": {k: round(v, 4) for k, v in z_scores.items()},
            "anomaly_detected": is_anomaly,
            "detection_reason": reason,
            "max_z_score": round(max_z, 4)
        })

    # Detection summary
    results["detection_summary"] = {
        "total_epochs": len(attack_indices),
        "detected_count": detected_count,
        "detection_rate": round(detected_count / len(attack_indices), 4) if attack_indices else 0
    }

    print(f"  Detection: {detected_count}/{len(attack_indices)} epochs flagged "
          f"({100*detected_count/len(attack_indices):.1f}%)")

    # Save results
    bp_s = format_numeric_token(baseline_pct)
    mp_s = format_numeric_token(malicious_pct)
    zt_s = format_numeric_token(z_threshold)
    output_file = os.path.join(output_dir, f"mi_{bp_s}_{mp_s}_{zt_s}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {os.path.basename(output_file)}")
    return results


def run_sparse_configuration(directory: str, baseline_pct: float, malicious_pct: float,
                             dataset_name: str, output_dir: str,
                             attack_prob: float = 0.3, window_size: int = 10,
                             z_threshold: float = 2.5) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*'))
                            if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_pct / 100))

    print(f"\n  [MI SPARSE] Baseline={baseline_pct}%, Mal={malicious_pct}%, "
          f"AttackProb={attack_prob}, Window={window_size}")

    baseline_indices = list(range(baseline_count))
    attack_pool = list(range(baseline_count, total_epochs))

    # Determine attacked windows
    windows = [attack_pool[i:i+window_size]
               for i in range(0, len(attack_pool), window_size)]
    attacked_epoch_set = set()

    for window in windows:
        if random.random() < attack_prob:
            attacked_epoch_set.update(window)

    print(f"  Windows: {len(windows)}, Attacked epochs: {len(attacked_epoch_set)}")

    # Phase 1: Build baseline
    print(f"  Phase 1: Building MI baseline...")
    baseline_metrics_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_metrics_list.append(calculate_mi_metrics(flows))

    baseline_stats = calculate_baseline_stats(baseline_metrics_list)
    print(f"  Baseline from {len(baseline_metrics_list)} epochs")

    # Phase 2: Analyze sparse attacks
    print(f"  Phase 2: Analyzing sparse attack pattern...")
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "MI Detector (Sparse)",
            "directory": directory,
            "total_epochs": total_epochs,
            "baseline_epoch_count": baseline_count,
            "baseline_percentage": baseline_pct,
            "malicious_percentage": malicious_pct,
            "sparse_mode": True,
            "sparse_attack_prob": attack_prob,
            "sparse_window_size": window_size,
            "z_threshold": z_threshold,
        },
        "baseline": {k: round(v, 6) if isinstance(v, float) else v
                    for k, v in baseline_stats.items()},
        "epochs": []
    }

    tp = fp = fn = tn = 0

    for idx in attack_pool:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]
        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        is_attacked = idx in attacked_epoch_set

        if is_attacked:
            num_malicious = int(len(clean_flows) * (malicious_pct / 100))
            combined_flows = clean_flows + generate_malicious_flows(num_malicious, clean_flows)
        else:
            num_malicious = 0
            combined_flows = clean_flows

        observed_metrics = calculate_mi_metrics(combined_flows)
        z_scores = calculate_z_scores(baseline_stats, observed_metrics)
        is_anomaly, reason, max_z = detect_anomaly(z_scores, z_threshold)

        # Confusion matrix
        if is_attacked and is_anomaly:
            tp += 1
        elif is_attacked and not is_anomaly:
            fn += 1
        elif not is_attacked and is_anomaly:
            fp += 1
        else:
            tn += 1

        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked_ground_truth": is_attacked,
            "anomaly_detected": is_anomaly,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "mi_metrics": metrics_to_dict(observed_metrics),
            "z_scores": {k: round(v, 4) for k, v in z_scores.items()},
            "detection_reason": reason,
            "max_z_score": round(max_z, 4)
        })
        

    # Summary
    results["detection_summary"] = {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        "accuracy": round((tp + tn) / (tp + fp + tn + fn), 4) if (tp + fp + tn + fn) > 0 else 0,
    }

    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Precision: {results['detection_summary']['precision']:.2%}, "
          f"Recall: {results['detection_summary']['recall']:.2%}")

    bp_s = format_numeric_token(baseline_pct)
    mp_s = format_numeric_token(malicious_pct)
    zt_s = format_numeric_token(z_threshold)
    output_file = os.path.join(output_dir, f"mi_{bp_s}_{mp_s}_{zt_s}_sparse.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {os.path.basename(output_file)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MI-based Attack Detector")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--attack-prob", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print(f"MUTUAL INFORMATION ATTACK DETECTOR [{'SPARSE' if args.sparse else 'FULL'}]")
    print("=" * 80)

    directory = args.directory or input("\nEpoch directory: ").strip()
    dataset_name = os.path.basename(os.path.normpath(directory))
    output_dir = os.path.join(directory, "json")
    os.makedirs(output_dir, exist_ok=True)

    BASELINE_PERCENTAGES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    MALICIOUS_PERCENTAGES = [1.0, 2.0, 3.0, 5.0, 10.0]
    Z_THRESHOLDS = [2.0, 2.5, 3.0]

    print(f"\nDataset: {dataset_name}")
    print(f"Baseline %: {BASELINE_PERCENTAGES}")
    print(f"Malicious %: {MALICIOUS_PERCENTAGES}")
    print(f"Z-thresholds: {Z_THRESHOLDS}")

    count = 0
    total = len(BASELINE_PERCENTAGES) * len(MALICIOUS_PERCENTAGES) * len(Z_THRESHOLDS)

    for bp in BASELINE_PERCENTAGES:
        for mp in MALICIOUS_PERCENTAGES:
            for zt in Z_THRESHOLDS:
                count += 1
                print(f"\n[{count}/{total}] Baseline={bp} Malicious={mp} Z={zt}", "-" * 40)
                try:
                    if args.sparse:
                        run_sparse_configuration(directory, bp, mp, dataset_name,
                                                output_dir, args.attack_prob,
                                                args.window_size, zt)
                    else:
                        run_single_configuration(directory, bp, mp, dataset_name,
                                               output_dir, zt)
                except Exception as e:
                    print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"COMPLETE! Results in {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
