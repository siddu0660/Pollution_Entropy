from typing import Tuple, Dict, List, Deque
from dataclasses import dataclass, field
import zlib
import hashlib
import math
import os
import random
import glob
import json
import statistics
from collections import deque
from datetime import datetime

# ML Imports
import numpy as np
from sklearn.ensemble import IsolationForest

@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    
    def get_signature(self) -> int:
        sig_str = f"{self.src_ip}{self.dst_ip}{self.src_port}{self.dst_port}{self.protocol}"
        return int(hashlib.sha256(sig_str.encode()).hexdigest(), 16)

def parse_flows(flow_file: str) -> List[Flow]:
    flows = []
    with open(flow_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) != 36:
                continue
            src_ip = line[0:12]
            dst_ip = line[12:24]
            src_port = line[24:29]
            dst_port = line[29:34]
            protocol = line[34:36]
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    return flows

def parse_flows_from_directory(directory: str) -> List[Flow]:
    all_flows = []
    file_pattern = os.path.join(directory, '*')
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Warning: No files found in directory: {directory}")
        return all_flows
    
    print(f"\nFound {len(files)} files in directory: {directory}")
    for file_path in sorted(files):
        if os.path.isfile(file_path):
            try:
                flows = parse_flows(file_path)
                all_flows.extend(flows)
                print(f"  Loaded {len(flows):,} flows from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Error reading {file_path}: {str(e)}")
    
    print(f"\nTotal flows loaded (epoch size): {len(all_flows):,}")
    return all_flows

# CMS
@dataclass
class CountMinSketch:
    width: int
    depth: int
    sketch: List[List[int]] = field(default_factory=list, init=False, repr=False)
    total_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("Width and depth must be positive integers")
        self.sketch = [[0 for _ in range(self.width)] for _ in range(self.depth)]
    
    def hash(self, item: str, seed: int) -> int:
        hash_input = f"{item}_{seed}".encode('utf-8')
        hash_value = zlib.crc32(hash_input)
        return hash_value % self.width
    
    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            col = self.hash(item, i)
            self.sketch[i][col] += count
        self.total_count += count
    
    def estimate(self, item: str) -> int:
        min_count = float('inf')
        for i in range(self.depth):
            col = self.hash(item, i)
            min_count = min(min_count, self.sketch[i][col])
        return int(min_count)
    
    def estimate_cardinality_mle(self) -> float:
        cardinality = 0.0
        for count in self.sketch[0]:
            if count > 0:
                contribution = self.width * (1 - math.exp(-count / self.width))
                cardinality += contribution
        return cardinality

# Entropy
@dataclass
class Entropy:
    width: int
    depth: int
    src_ip_cms: CountMinSketch = field(init=False)
    dst_ip_cms: CountMinSketch = field(init=False)
    src_port_cms: CountMinSketch = field(init=False)
    dst_port_cms: CountMinSketch = field(init=False)
    src_dst_ip_cms: CountMinSketch = field(init=False)
    src_dst_port_cms: CountMinSketch = field(init=False)
    
    def __post_init__(self):
        self.src_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.src_port_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_port_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.src_dst_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.src_dst_port_cms = CountMinSketch(width=self.width, depth=self.depth)
    
    def add_flow(self, flow: Flow) -> None:
        self.src_ip_cms.add(flow.src_ip)
        self.dst_ip_cms.add(flow.dst_ip)
        self.src_port_cms.add(flow.src_port)
        self.dst_port_cms.add(flow.dst_port)
        self.src_dst_ip_cms.add(flow.src_ip + flow.dst_ip)
        self.src_dst_port_cms.add(flow.src_port + flow.dst_port)
    
    def calculate_entropy(self, cms: CountMinSketch, items: List[str]) -> Tuple[float, float, int]:
        if cms.total_count == 0:
            return 0.0, 0.0, 0
        
        entropy = 0.0
        non_zero_items = 0
        
        for item in items:
            freq = cms.estimate(item)
            if freq > 0:
                non_zero_items += 1
                prob = freq / cms.total_count
                entropy -= prob * math.log2(prob)
        
        cardinality = cms.estimate_cardinality_mle()
        return entropy, cardinality, non_zero_items
    
    def get_src_ip_entropy(self, unique_src_ips: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_ip_cms, unique_src_ips)
    def get_dst_ip_entropy(self, unique_dst_ips: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.dst_ip_cms, unique_dst_ips)
    def get_src_port_entropy(self, unique_src_ports: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_port_cms, unique_src_ports)
    def get_dst_port_entropy(self, unique_dst_ports: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.dst_port_cms, unique_dst_ports)
    def get_src_dst_ip_entropy(self, unique_pairs: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_dst_ip_cms, unique_pairs)
    def get_src_dst_port_entropy(self, unique_pairs: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_dst_port_cms, unique_pairs)

    @staticmethod
    def calculate_effectiveness(entropy: float, cardinality: float) -> float:
        if cardinality == 0: return 0.0
        return math.pow(2, entropy) / cardinality
    
@dataclass
class EntropyMetrics:
    total_flows: int
    unique_src_ips: int
    unique_dst_ips: int
    unique_src_ports: int
    unique_dst_ports: int
    src_ip_entropy: float
    src_ip_cardinality: float
    dst_ip_entropy: float
    dst_ip_cardinality: float
    src_port_entropy: float
    src_port_cardinality: float
    dst_port_entropy: float
    dst_port_cardinality: float
    src_ip_uniformity: float
    dst_ip_uniformity: float
    src_dst_ratio: float
    src_dst_ip_entropy: float
    src_dst_ip_cardinality: float
    src_dst_port_entropy: float
    src_dst_port_cardinality: float
    # New Mutual Information Metrics
    ip_mutual_information: float
    port_mutual_information: float

def calculate_all_entropies(flows: List[Flow], width: int = 1024, depth: int = 5) -> EntropyMetrics:
    entropy_calc = Entropy(width=width, depth=depth)
    for flow in flows:
        entropy_calc.add_flow(flow)
    
    unique_src_ips        = list(set(flow.src_ip for flow in flows))
    unique_dst_ips        = list(set(flow.dst_ip for flow in flows))
    unique_src_ports      = list(set(flow.src_port for flow in flows))
    unique_dst_ports      = list(set(flow.dst_port for flow in flows))
    unique_src_dst_ips    = list(set(flow.src_ip + flow.dst_ip for flow in flows))
    unique_src_dst_ports  = list(set(flow.src_port + flow.dst_port for flow in flows))
    
    src_ip_entropy,       src_ip_cardinality,       _ = entropy_calc.get_src_ip_entropy(unique_src_ips)
    dst_ip_entropy,       dst_ip_cardinality,       _ = entropy_calc.get_dst_ip_entropy(unique_dst_ips)
    src_port_entropy,     src_port_cardinality,     _ = entropy_calc.get_src_port_entropy(unique_src_ports)
    dst_port_entropy,     dst_port_cardinality,     _ = entropy_calc.get_dst_port_entropy(unique_dst_ports)
    src_dst_ip_entropy,   src_dst_ip_cardinality,   _ = entropy_calc.get_src_dst_ip_entropy(unique_src_dst_ips)
    src_dst_port_entropy, src_dst_port_cardinality, _ = entropy_calc.get_src_dst_port_entropy(unique_src_dst_ports)
    
    src_ip_uniformity = entropy_calc.calculate_effectiveness(src_ip_entropy, src_ip_cardinality)
    dst_ip_uniformity = entropy_calc.calculate_effectiveness(dst_ip_entropy, dst_ip_cardinality)
    
    # Calculate Mutual Information
    ip_mutual_information = src_ip_entropy + dst_ip_entropy - src_dst_ip_entropy
    port_mutual_information = src_port_entropy + dst_port_entropy - src_dst_port_entropy

    return EntropyMetrics(
        total_flows=len(flows),
        unique_src_ips=len(unique_src_ips),
        unique_dst_ips=len(unique_dst_ips),
        unique_src_ports=len(unique_src_ports),
        unique_dst_ports=len(unique_dst_ports),
        src_ip_entropy=src_ip_entropy,
        src_ip_cardinality=src_ip_cardinality,
        dst_ip_entropy=dst_ip_entropy,
        dst_ip_cardinality=dst_ip_cardinality,
        src_port_entropy=src_port_entropy,
        src_port_cardinality=src_port_cardinality,
        dst_port_entropy=dst_port_entropy,
        dst_port_cardinality=dst_port_cardinality,
        src_ip_uniformity=src_ip_uniformity,
        dst_ip_uniformity=dst_ip_uniformity,
        src_dst_ratio=src_ip_cardinality / dst_ip_cardinality if dst_ip_cardinality > 0 else 0,
        src_dst_ip_entropy=src_dst_ip_entropy,
        src_dst_ip_cardinality=src_dst_ip_cardinality,
        src_dst_port_entropy=src_dst_port_entropy,
        src_dst_port_cardinality=src_dst_port_cardinality,
        ip_mutual_information=ip_mutual_information,
        port_mutual_information=port_mutual_information
    )

def generate_malicious_flows(num_flows: int, base_flows: List[Flow]) -> List[Flow]:
    flows = []
    for i in range(num_flows):
        src_ip = f"{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}"
        dst_ip = f"{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}"
        src_port = f"{random.randint(0, 65535):05d}"
        dst_port_val = random.choice([80, 443, 808, 22, 53, 330, 543, random.randint(0, 65535)])
        dst_port = f"{dst_port_val % 65535:05d}"
        protocol = f"{random.choice([6, 17]):02d}"
        flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    return flows

def metrics_to_dict(metrics: EntropyMetrics) -> dict:
    return {
        "total_flows": metrics.total_flows,
        "unique_src_ips": metrics.unique_src_ips,
        "unique_dst_ips": metrics.unique_dst_ips,
        "unique_src_ports": metrics.unique_src_ports,
        "unique_dst_ports": metrics.unique_dst_ports,
        "src_ip_entropy": round(metrics.src_ip_entropy, 6),
        "dst_ip_entropy": round(metrics.dst_ip_entropy, 6),
        "src_port_entropy": round(metrics.src_port_entropy, 6),
        "dst_port_entropy": round(metrics.dst_port_entropy, 6),
        "src_ip_cardinality": round(metrics.src_ip_cardinality, 4),
        "dst_ip_cardinality": round(metrics.dst_ip_cardinality, 4),
        "src_port_cardinality": round(metrics.src_port_cardinality, 4),
        "dst_port_cardinality": round(metrics.dst_port_cardinality, 4),
        "src_ip_uniformity": round(metrics.src_ip_uniformity, 6),
        "dst_ip_uniformity": round(metrics.dst_ip_uniformity, 6),
        "src_dst_ratio": round(metrics.src_dst_ratio, 6),
        "src_dst_ip_entropy": round(metrics.src_dst_ip_entropy, 6),
        "src_dst_ip_cardinality": round(metrics.src_dst_ip_cardinality, 4),
        "src_dst_port_entropy": round(metrics.src_dst_port_entropy, 6),
        "src_dst_port_cardinality": round(metrics.src_dst_port_cardinality, 4),
        "ip_mutual_information": round(metrics.ip_mutual_information, 6),
        "port_mutual_information": round(metrics.port_mutual_information, 6)
    }

def calculate_baseline_entropies(baseline_metrics: List[EntropyMetrics]) -> Dict[str, float]:
    if not baseline_metrics: return {}
    return {
        "total_flows": max([m.total_flows for m in baseline_metrics]),
        "unique_src_ips": max([m.unique_src_ips for m in baseline_metrics]),
        "unique_dst_ips": max([m.unique_dst_ips for m in baseline_metrics]),
        "unique_src_ports": max([m.unique_src_ports for m in baseline_metrics]),
        "unique_dst_ports": max([m.unique_dst_ports for m in baseline_metrics]),
        "src_ip_entropy": max([m.src_ip_entropy for m in baseline_metrics]),
        "src_ip_cardinality": max([m.src_ip_cardinality for m in baseline_metrics]),
        "dst_ip_entropy": max([m.dst_ip_entropy for m in baseline_metrics]),
        "dst_ip_cardinality": max([m.dst_ip_cardinality for m in baseline_metrics]),
        "src_port_entropy": max([m.src_port_entropy for m in baseline_metrics]),
        "src_port_cardinality": max([m.src_port_cardinality for m in baseline_metrics]),
        "dst_port_entropy": max([m.dst_port_entropy for m in baseline_metrics]),
        "dst_port_cardinality": max([m.dst_port_cardinality for m in baseline_metrics]),
        "src_ip_uniformity": max([m.src_ip_uniformity for m in baseline_metrics]),
        "dst_ip_uniformity": max([m.dst_ip_uniformity for m in baseline_metrics]),
        "src_dst_ratio": max([m.src_dst_ratio for m in baseline_metrics]),
        "src_dst_ip_entropy": max([m.src_dst_ip_entropy for m in baseline_metrics]),
        "src_dst_ip_cardinality": max([m.src_dst_ip_cardinality for m in baseline_metrics]),
        "src_dst_port_entropy": max([m.src_dst_port_entropy for m in baseline_metrics]),
        "src_dst_port_cardinality": max([m.src_dst_port_cardinality for m in baseline_metrics]),
        "ip_mutual_information": max([m.ip_mutual_information for m in baseline_metrics]),
        "port_mutual_information": max([m.port_mutual_information for m in baseline_metrics])
    }

# ----------------- ML Feature Extraction Helper -----------------
ML_FEATURE_KEYS = [
    'src_ip_entropy', 'dst_ip_entropy', 'src_port_entropy', 'dst_port_entropy',
    'ip_mutual_information', 'port_mutual_information',
    'src_dst_ratio', 'src_ip_cardinality', 'dst_ip_cardinality'
]

def extract_features(metrics: EntropyMetrics) -> List[float]:
    """Converts a metrics dataclass into a fixed float array for the ML model."""
    return [getattr(metrics, key) for key in ML_FEATURE_KEYS]

# ----------------------------------------------------------------

def run_sparse_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str,
                             attack_prob: float = 0.3, window_size: int = 10) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  [SPARSE] Configuration: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%,"
          f" AttackProb={attack_prob}, WindowSize={window_size}")

    baseline_indices = list(range(baseline_count))
    attack_pool = list(range(baseline_count, total_epochs))

    windows = []
    for w_start in range(0, len(attack_pool), window_size):
        windows.append(attack_pool[w_start: w_start + window_size])

    attacked_windows = []
    for w_idx, window in enumerate(windows):
        if random.random() < attack_prob:
            attacked_windows.append(w_idx)

    attacked_epoch_set = set()
    for w_idx in attacked_windows:
        for ep_idx in windows[w_idx]:
            attacked_epoch_set.add(ep_idx)

    print(f"  Phase 1: Calculating baseline entropies & Training Isolation Forest...")
    baseline_metrics_list = []
    baseline_feature_matrix = []
    
    for idx in baseline_indices:
        epoch_file = epoch_files[idx]
        flows = parse_flows(epoch_file)
        if not flows: continue
        metrics = calculate_all_entropies(flows)
        baseline_metrics_list.append(metrics)
        baseline_feature_matrix.append(extract_features(metrics))

    baseline = calculate_baseline_entropies(baseline_metrics_list)
    
    # Train the Isolation Forest ONLY on the clean baseline data
    clf = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
    if baseline_feature_matrix:
        clf.fit(baseline_feature_matrix)
        print(f"  Isolation Forest trained on {len(baseline_feature_matrix)} clean epochs.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "sparse_mode": True,
            "ml_model": "IsolationForest",
            "ml_features_used": ML_FEATURE_KEYS
        },
        "baseline": {k: round(v, 6) if isinstance(v, float) else v for k, v in baseline.items()},
        "epochs": []
    }

    print(f"  Phase 2: Analyzing epochs with ML inference...")
    for idx in attack_pool:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]
        clean_flows = parse_flows(epoch_file)
        if not clean_flows: continue

        is_attacked = idx in attacked_epoch_set

        if is_attacked:
            num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
            malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
            combined_flows = clean_flows + malicious_flows
            observed_metrics = calculate_all_entropies(combined_flows)
        else:
            num_malicious = 0
            combined_flows = clean_flows
            observed_metrics = calculate_all_entropies(clean_flows)

        # Generate ML prediction based strictly on the current window's metrics
        test_vector = [extract_features(observed_metrics)]
        prediction = clf.predict(test_vector)[0] if baseline_feature_matrix else 1
        anomaly_score = clf.decision_function(test_vector)[0] if baseline_feature_matrix else 0.0

        epoch_result = {
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked_ground_truth": is_attacked,
            "ml_prediction_anomaly": bool(prediction == -1), # True if model flagged it
            "ml_anomaly_score": round(float(anomaly_score), 6),
            "total_flows_count": len(combined_flows),
            "observed_metrics": metrics_to_dict(observed_metrics)
        }
        results["epochs"].append(epoch_result)

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_ml_sparse.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    return results

def run_single_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str) -> dict:
    # Included for completeness (Full Attack Mode). Follows the same ML logic as Sparse.
    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))
    baseline_indices = list(range(baseline_count))
    attack_indices = list(range(baseline_count, total_epochs))

    print(f"  Phase 1: Calculating baseline entropies & Training Isolation Forest...")
    baseline_metrics_list = []
    baseline_feature_matrix = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if not flows: continue
        metrics = calculate_all_entropies(flows)
        baseline_metrics_list.append(metrics)
        baseline_feature_matrix.append(extract_features(metrics))

    baseline = calculate_baseline_entropies(baseline_metrics_list)
    
    clf = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
    if baseline_feature_matrix:
        clf.fit(baseline_feature_matrix)

    results = {
        "metadata": {"dataset_name": dataset_name, "sparse_mode": False},
        "baseline": {k: round(v, 6) if isinstance(v, float) else v for k, v in baseline.items()},
        "epochs": []
    }

    print(f"  Phase 2: Analyzing attack epochs with ML inference...")
    for idx in attack_indices:
        epoch_file = epoch_files[idx]
        clean_flows = parse_flows(epoch_file)
        if not clean_flows: continue

        num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
        combined_flows = clean_flows + generate_malicious_flows(num_malicious, clean_flows)
        malicious_metrics = calculate_all_entropies(combined_flows)

        test_vector = [extract_features(malicious_metrics)]
        prediction = clf.predict(test_vector)[0] if baseline_feature_matrix else 1
        anomaly_score = clf.decision_function(test_vector)[0] if baseline_feature_matrix else 0.0

        results["epochs"].append({
            "epoch_index": idx,
            "attacked_ground_truth": True,
            "ml_prediction_anomaly": bool(prediction == -1),
            "ml_anomaly_score": round(float(anomaly_score), 6),
            "observed_metrics": metrics_to_dict(malicious_metrics)
        })

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_ml.json"
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ML-Augmented Entropy Analysis")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--attack-prob", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()

    directory = args.directory if args.directory else input("\nEnter directory path: ").strip()
    dataset_name = os.path.basename(os.path.normpath(directory))
    json_output_dir = os.path.join(directory, "json")
    os.makedirs(json_output_dir, exist_ok=True)

    BASELINE_PERCENTAGES = [30.0]
    MALICIOUS_PERCENTAGES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    for b_pct in BASELINE_PERCENTAGES:
        for m_pct in MALICIOUS_PERCENTAGES:
            if args.sparse:
                run_sparse_configuration(directory, b_pct, m_pct, dataset_name, json_output_dir, args.attack_prob, args.window_size)
            else:
                run_single_configuration(directory, b_pct, m_pct, dataset_name, json_output_dir)

if __name__ == "__main__":
    main()