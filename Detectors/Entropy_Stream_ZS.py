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


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------
@dataclass
class Entropy:
    width: int
    depth: int
    src_ip_cms: CountMinSketch = field(init=False)
    dst_ip_cms: CountMinSketch = field(init=False)
    src_port_cms: CountMinSketch = field(init=False)
    dst_port_cms: CountMinSketch = field(init=False)
    
    def __post_init__(self):
        self.src_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.src_port_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_port_cms = CountMinSketch(width=self.width, depth=self.depth)
    
    def add_flow(self, flow: Flow) -> None:
        self.src_ip_cms.add(flow.src_ip)
        self.dst_ip_cms.add(flow.dst_ip)
        self.src_port_cms.add(flow.src_port)
        self.dst_port_cms.add(flow.dst_port)
    
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

    @staticmethod
    def calculate_effectiveness(entropy: float, cardinality: float) -> float:
        if cardinality == 0:
            return 0.0
        return math.pow(2, entropy) / cardinality

    @staticmethod
    def normalized_entropy(entropy: float, cardinality: float) -> float:
        """Entropy divided by log2(cardinality). Bounded [0,1].
        Attack traffic (uniform, high-cardinality) pushes this toward 1.0."""
        if cardinality <= 1:
            return 0.0
        max_h = math.log2(cardinality)
        return entropy / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# EntropyMetrics
# ---------------------------------------------------------------------------
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
    # New: normalized entropy (bounded [0,1])
    src_ip_norm_entropy: float = 0.0
    dst_ip_norm_entropy: float = 0.0
    src_port_norm_entropy: float = 0.0
    dst_port_norm_entropy: float = 0.0


def calculate_all_entropies(flows: List[Flow], width: int = 1024, depth: int = 5) -> EntropyMetrics:
    entropy_calc = Entropy(width=width, depth=depth)
    for flow in flows:
        entropy_calc.add_flow(flow)
    
    unique_src_ips   = list(set(flow.src_ip   for flow in flows))
    unique_dst_ips   = list(set(flow.dst_ip   for flow in flows))
    unique_src_ports = list(set(flow.src_port for flow in flows))
    unique_dst_ports = list(set(flow.dst_port for flow in flows))
    
    src_ip_entropy,   src_ip_cardinality,   _ = entropy_calc.get_src_ip_entropy(unique_src_ips)
    dst_ip_entropy,   dst_ip_cardinality,   _ = entropy_calc.get_dst_ip_entropy(unique_dst_ips)
    src_port_entropy, src_port_cardinality, _ = entropy_calc.get_src_port_entropy(unique_src_ports)
    dst_port_entropy, dst_port_cardinality, _ = entropy_calc.get_dst_port_entropy(unique_dst_ports)
    
    src_ip_uniformity = entropy_calc.calculate_effectiveness(src_ip_entropy, src_ip_cardinality)
    dst_ip_uniformity = entropy_calc.calculate_effectiveness(dst_ip_entropy, dst_ip_cardinality)

    src_ip_norm   = Entropy.normalized_entropy(src_ip_entropy,   src_ip_cardinality)
    dst_ip_norm   = Entropy.normalized_entropy(dst_ip_entropy,   dst_ip_cardinality)
    src_port_norm = Entropy.normalized_entropy(src_port_entropy, src_port_cardinality)
    dst_port_norm = Entropy.normalized_entropy(dst_port_entropy, dst_port_cardinality)
    
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
        src_ip_norm_entropy=src_ip_norm,
        dst_ip_norm_entropy=dst_ip_norm,
        src_port_norm_entropy=src_port_norm,
        dst_port_norm_entropy=dst_port_norm,
    )


# ---------------------------------------------------------------------------
# Malicious flow generation
# ---------------------------------------------------------------------------
def generate_malicious_flows(num_flows: int, base_flows: List[Flow]) -> List[Flow]:
    flows = []
    for i in range(num_flows):
        src_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        dst_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        src_port = f"{random.randint(0, 65535):05d}"
        dst_port_val = random.choice([80, 443, 808, 22, 53, 330, 543, random.randint(0, 65535)])
        dst_port = f"{dst_port_val % 65535:05d}"
        protocol = f"{random.choice([6, 17]):02d}"
        flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    return flows


# ---------------------------------------------------------------------------
# Metrics → dict
# ---------------------------------------------------------------------------
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
        "src_ip_norm_entropy": round(metrics.src_ip_norm_entropy, 6),
        "dst_ip_norm_entropy": round(metrics.dst_ip_norm_entropy, 6),
        "src_port_norm_entropy": round(metrics.src_port_norm_entropy, 6),
        "dst_port_norm_entropy": round(metrics.dst_port_norm_entropy, 6),
    }


# ---------------------------------------------------------------------------
# Baseline: mean + std per feature  (NEW — previously only mean was stored)
# ---------------------------------------------------------------------------

# All scalar fields we track in the baseline
_BASELINE_FIELDS = [
    "total_flows",
    "unique_src_ips", "unique_dst_ips", "unique_src_ports", "unique_dst_ports",
    "src_ip_entropy", "src_ip_cardinality",
    "dst_ip_entropy", "dst_ip_cardinality",
    "src_port_entropy", "src_port_cardinality",
    "dst_port_entropy", "dst_port_cardinality",
    "src_ip_uniformity", "dst_ip_uniformity",
    "src_dst_ratio",
    "src_ip_norm_entropy", "dst_ip_norm_entropy",
    "src_port_norm_entropy", "dst_port_norm_entropy",
]


def calculate_baseline_entropies(baseline_metrics: List[EntropyMetrics]) -> Dict[str, float]:
    """Return mean and std for every feature across all baseline epochs."""
    if not baseline_metrics:
        return {}
    baseline: Dict[str, float] = {}
    for f in _BASELINE_FIELDS:
        vals = [getattr(m, f) for m in baseline_metrics]
        baseline[f"{f}_mean"] = statistics.mean(vals)
        baseline[f"{f}_std"]  = statistics.stdev(vals) if len(vals) > 1 else 1e-9
    return baseline


# ---------------------------------------------------------------------------
# Z-score computation  (NEW)
# ---------------------------------------------------------------------------

def calculate_z_scores(baseline: Dict[str, float], current: EntropyMetrics) -> Dict[str, float]:
    """Compute per-feature Z-scores: (value - baseline_mean) / baseline_std."""
    z: Dict[str, float] = {}
    for f in _BASELINE_FIELDS:
        mean = baseline.get(f"{f}_mean", 0.0)
        std  = baseline.get(f"{f}_std",  1e-9) or 1e-9
        z[f"{f}_z"] = (getattr(current, f) - mean) / std
    return z


def anomaly_score(z_scores: Dict[str, float]) -> float:
    """RMS of all Z-scores — a single composite anomaly magnitude."""
    vals = list(z_scores.values())
    if not vals:
        return 0.0
    return math.sqrt(statistics.mean(v ** 2 for v in vals))


# Key features whose individual Z-scores we check for threshold breaches
_KEY_FEATURES = [
    "src_ip_entropy_z",
    "dst_ip_entropy_z",
    "src_ip_cardinality_z",
    "dst_ip_cardinality_z",
    "total_flows_z",
    "src_ip_norm_entropy_z",
    "dst_ip_norm_entropy_z",
]


def detect_attack(z_scores: Dict[str, float],
                  flow_ratio: float,
                  threshold_z: float = 2.5,
                  threshold_composite: float = 2.0,
                  threshold_flow_ratio: float = 1.05) -> Tuple[bool, str]:
    """
    Binary decision + reason string.

    Rules (any one is enough):
      1. Any key-feature |Z| > threshold_z
      2. Composite anomaly score > threshold_composite
      3. Flow-volume ratio > threshold_flow_ratio
    """
    for feat in _KEY_FEATURES:
        z = z_scores.get(feat, 0.0)
        if abs(z) > threshold_z:
            return True, f"key_feature_breach: {feat}={z:.3f}"

    score = anomaly_score(z_scores)
    if score > threshold_composite:
        return True, f"composite_score: {score:.3f}"

    if flow_ratio > threshold_flow_ratio:
        return True, f"flow_ratio: {flow_ratio:.3f}"

    return False, "clean"


# ---------------------------------------------------------------------------
# Legacy helper kept for compatibility
# ---------------------------------------------------------------------------
def calculate_entropy_changes(baseline: Dict[str, float], current: EntropyMetrics) -> Dict[str, float]:
    """Raw delta (epoch value − baseline mean) for every feature."""
    changes: Dict[str, float] = {}
    for f in _BASELINE_FIELDS:
        mean = baseline.get(f"{f}_mean", baseline.get(f, 0.0))
        changes[f"{f}_change"] = getattr(current, f) - mean
    return changes


# ---------------------------------------------------------------------------
# Z-score + detection block for JSON output  (NEW helper)
# ---------------------------------------------------------------------------
def build_detection_block(baseline: Dict[str, float],
                           observed_metrics: EntropyMetrics,
                           baseline_total_flows: float,
                           threshold_z: float = 2.5,
                           threshold_composite: float = 2.0,
                           threshold_flow_ratio: float = 1.05) -> dict:
    z_scores   = calculate_z_scores(baseline, observed_metrics)
    comp_score = anomaly_score(z_scores)
    flow_ratio = (observed_metrics.total_flows / baseline_total_flows
                  if baseline_total_flows > 0 else 1.0)
    flagged, reason = detect_attack(z_scores, flow_ratio,
                                    threshold_z, threshold_composite,
                                    threshold_flow_ratio)
    return {
        "z_scores":        {k: round(v, 4) for k, v in z_scores.items()},
        "anomaly_score":   round(comp_score, 4),
        "flow_ratio":      round(flow_ratio, 4),
        "flagged":         flagged,
        "flag_reason":     reason,
        "thresholds_used": {
            "z_threshold":          threshold_z,
            "composite_threshold":  threshold_composite,
            "flow_ratio_threshold": threshold_flow_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Single (full-attack) configuration
# ---------------------------------------------------------------------------
def run_single_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs   = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  Configuration: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%")
    print(f"  Total epochs: {total_epochs}, Baseline epochs: {baseline_count}")

    baseline_indices = list(range(baseline_count))
    attack_indices   = list(range(baseline_count, total_epochs))

    # Phase 1 — baseline (now stores mean + std)
    print(f"  Phase 1: Calculating baseline entropies...")
    baseline_metrics_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_metrics_list.append(calculate_all_entropies(flows))

    baseline = calculate_baseline_entropies(baseline_metrics_list)
    baseline_mean_flows = baseline.get("total_flows_mean", 1.0)
    print(f"  Baseline established from {len(baseline_metrics_list)} epochs.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "directory": directory,
            "total_epochs": total_epochs,
            "baseline_epoch_count": baseline_count,
            "baseline_percentage": baseline_percentage,
            "malicious_percentage": malicious_percentage,
            "baseline_indices": baseline_indices,
            "attack_indices": attack_indices,
            "sparse_mode": False,
        },
        "baseline": {k: round(v, 6) if isinstance(v, float) else v for k, v in baseline.items()},
        "epochs": [],
    }

    # Phase 2
    print(f"  Phase 2: Analyzing attack epochs...")
    for idx in attack_indices:
        epoch_file  = epoch_files[idx]
        epoch_name  = os.path.splitext(os.path.basename(epoch_file))[0]
        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        clean_metrics  = calculate_all_entropies(clean_flows)
        num_malicious  = int(len(clean_flows) * (malicious_percentage / 100))
        malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
        combined_flows  = clean_flows + malicious_flows
        observed_metrics = calculate_all_entropies(combined_flows)

        changes   = calculate_entropy_changes(baseline, observed_metrics)
        detection = build_detection_block(baseline, observed_metrics, baseline_mean_flows)

        results["epochs"].append({
            "epoch_index":         idx,
            "epoch_name":          epoch_name,
            "attacked":            True,
            "clean_flows_count":   len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count":   len(combined_flows),
            "clean_metrics":       metrics_to_dict(clean_metrics),
            "malicious_metrics":   metrics_to_dict(observed_metrics),
            "entropy_changes":     {k: round(v, 6) if isinstance(v, float) else v for k, v in changes.items()},
            "detection":           detection,
        })

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")
    return results


# ---------------------------------------------------------------------------
# Sparse configuration
# ---------------------------------------------------------------------------
def run_sparse_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str,
                             attack_prob: float = 0.3, window_size: int = 10) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs   = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  [SPARSE] Configuration: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%,"
          f" AttackProb={attack_prob}, WindowSize={window_size}")
    print(f"  Total epochs: {total_epochs}, Baseline epochs: {baseline_count}")

    baseline_indices = list(range(baseline_count))
    attack_pool      = list(range(baseline_count, total_epochs))

    # Build windows
    windows = []
    for w_start in range(0, len(attack_pool), window_size):
        windows.append(attack_pool[w_start: w_start + window_size])

    attacked_windows, clean_windows = [], []
    for w_idx, window in enumerate(windows):
        if random.random() < attack_prob:
            attacked_windows.append(w_idx)
        else:
            clean_windows.append(w_idx)

    attacked_epoch_set = set()
    for w_idx in attacked_windows:
        for ep_idx in windows[w_idx]:
            attacked_epoch_set.add(ep_idx)

    attacked_window_info = [
        {
            "window_index":  w_idx,
            "epoch_indices": windows[w_idx],
            "epoch_range":   [windows[w_idx][0], windows[w_idx][-1]] if windows[w_idx] else [],
        }
        for w_idx in attacked_windows
    ]

    print(f"  Windows total: {len(windows)}, Windows attacked: {len(attacked_windows)}, "
          f"Windows clean: {len(clean_windows)}")
    if attacked_window_info:
        ranges = [f"[{w['epoch_range'][0]}-{w['epoch_range'][1]}]" for w in attacked_window_info]
        print(f"  Attacked epoch windows: {', '.join(ranges)}")
    else:
        print(f"  No windows selected for attack this run.")

    # Phase 1 — baseline (mean + std)
    print(f"  Phase 1: Calculating baseline entropies...")
    baseline_metrics_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_metrics_list.append(calculate_all_entropies(flows))

    baseline = calculate_baseline_entropies(baseline_metrics_list)
    baseline_mean_flows = baseline.get("total_flows_mean", 1.0)
    print(f"  Baseline established from {len(baseline_metrics_list)} epochs.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "directory": directory,
            "total_epochs": total_epochs,
            "baseline_epoch_count": baseline_count,
            "baseline_percentage": baseline_percentage,
            "malicious_percentage": malicious_percentage,
            "baseline_indices": baseline_indices,
            "attack_indices": attack_pool,
            "sparse_mode": True,
            "sparse_attack_prob": attack_prob,
            "sparse_window_size": window_size,
            "total_windows": len(windows),
            "attacked_windows": attacked_window_info,
            "attacked_window_count": len(attacked_windows),
            "clean_window_count": len(clean_windows),
        },
        "baseline": {k: round(v, 6) if isinstance(v, float) else v for k, v in baseline.items()},
        "epochs": [],
    }

    # Phase 2
    print(f"  Phase 2: Analyzing epochs (sparse attack)...")
    flagged_count  = 0
    correct_flags  = 0
    false_positives = 0

    for idx in attack_pool:
        epoch_file  = epoch_files[idx]
        epoch_name  = os.path.splitext(os.path.basename(epoch_file))[0]
        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        is_attacked  = idx in attacked_epoch_set
        clean_metrics = calculate_all_entropies(clean_flows)

        if is_attacked:
            num_malicious   = int(len(clean_flows) * (malicious_percentage / 100))
            malicious_flows  = generate_malicious_flows(num_malicious, clean_flows)
            combined_flows   = clean_flows + malicious_flows
            observed_metrics = calculate_all_entropies(combined_flows)
        else:
            num_malicious    = 0
            combined_flows   = clean_flows
            observed_metrics = clean_metrics

        changes   = calculate_entropy_changes(baseline, observed_metrics)
        detection = build_detection_block(baseline, observed_metrics, baseline_mean_flows)

        # Running accuracy counters
        if detection["flagged"]:
            flagged_count += 1
            if is_attacked:
                correct_flags += 1
            else:
                false_positives += 1

        results["epochs"].append({
            "epoch_index":           idx,
            "epoch_name":            epoch_name,
            "attacked":              is_attacked,
            "clean_flows_count":     len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count":     len(combined_flows),
            "clean_metrics":         metrics_to_dict(clean_metrics),
            "malicious_metrics":     metrics_to_dict(observed_metrics),
            "entropy_changes":       {k: round(v, 6) if isinstance(v, float) else v for k, v in changes.items()},
            "detection":             detection,
        })

    # Summary stats
    total_attacked = sum(1 for e in results["epochs"] if e["attacked"])
    total_clean    = len(results["epochs"]) - total_attacked
    missed         = total_attacked - correct_flags

    results["detection_summary"] = {
        "total_epochs_analyzed": len(results["epochs"]),
        "truly_attacked":   total_attacked,
        "truly_clean":      total_clean,
        "flagged_total":    flagged_count,
        "true_positives":   correct_flags,
        "false_positives":  false_positives,
        "false_negatives":  missed,
        "detection_rate":   round(correct_flags / total_attacked, 4) if total_attacked > 0 else None,
        "false_positive_rate": round(false_positives / total_clean, 4) if total_clean > 0 else None,
    }

    print(f"\n  === DETECTION SUMMARY ===")
    print(f"  Truly attacked epochs : {total_attacked}")
    print(f"  True positives        : {correct_flags}")
    print(f"  False positives       : {false_positives}")
    print(f"  False negatives       : {missed}")
    if total_attacked > 0:
        print(f"  Detection rate        : {correct_flags/total_attacked:.1%}")
    if total_clean > 0:
        print(f"  False positive rate   : {false_positives/total_clean:.1%}")

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_sparse.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Entropy Analysis System - Multi-Configuration (with Z-score detection)"
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Directory containing epoch flow files")
    parser.add_argument("--sparse", action="store_true",
                        help="Sparse attack mode: attack only randomly selected windows of epochs")
    parser.add_argument("--attack-prob", type=float, default=0.3,
                        help="Probability of selecting each window for attack in sparse mode (default: 0.3)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Number of epochs per attack window in sparse mode (default: 10)")
    parser.add_argument("--threshold-z", type=float, default=2.5,
                        help="Z-score threshold for single-feature flagging (default: 2.5)")
    parser.add_argument("--threshold-composite", type=float, default=2.0,
                        help="Composite anomaly score threshold (default: 2.0)")
    parser.add_argument("--threshold-flow-ratio", type=float, default=1.05,
                        help="Flow volume ratio threshold vs baseline mean (default: 1.05)")
    args = parser.parse_args()

    print("=" * 80)
    mode_label = "SPARSE" if args.sparse else "FULL"
    print(f"BASELINE ENTROPY ANALYSIS SYSTEM - MULTI-CONFIGURATION [{mode_label} MODE]")
    print("=" * 80)

    if args.directory:
        directory = args.directory
    else:
        directory = input("\nEnter the directory path containing epoch flow files: ").strip()

    dataset_name = os.path.basename(os.path.normpath(directory))

    json_output_dir = os.path.join(directory, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    print(f"\nOutput directory created: {json_output_dir}")

    BASELINE_PERCENTAGES  = [5.0, 10.0]
    MALICIOUS_PERCENTAGES = [1.0, 5.0, 10.0, 20.0]

    print(f"\nDataset: {dataset_name}")
    if args.sparse:
        print(f"Mode: SPARSE  |  Attack probability: {args.attack_prob}  |  Window size: {args.window_size} epochs")
    print(f"Detection thresholds — Z: {args.threshold_z}, Composite: {args.threshold_composite}, "
          f"Flow ratio: {args.threshold_flow_ratio}")
    print(f"Baseline percentages to test: {BASELINE_PERCENTAGES}")
    print(f"Malicious percentages to test: {MALICIOUS_PERCENTAGES}")
    print(f"Total configurations: {len(BASELINE_PERCENTAGES) * len(MALICIOUS_PERCENTAGES)}")
    print("-" * 80)

    config_count  = 0
    total_configs = len(BASELINE_PERCENTAGES) * len(MALICIOUS_PERCENTAGES)

    for baseline_pct in BASELINE_PERCENTAGES:
        for malicious_pct in MALICIOUS_PERCENTAGES:
            config_count += 1
            print(f"\n[Configuration {config_count}/{total_configs}]")
            print("-" * 80)
            try:
                if args.sparse:
                    run_sparse_configuration(
                        directory=directory,
                        baseline_percentage=baseline_pct,
                        malicious_percentage=malicious_pct,
                        dataset_name=dataset_name,
                        output_dir=json_output_dir,
                        attack_prob=args.attack_prob,
                        window_size=args.window_size,
                    )
                else:
                    run_single_configuration(
                        directory=directory,
                        baseline_percentage=baseline_pct,
                        malicious_percentage=malicious_pct,
                        dataset_name=dataset_name,
                        output_dir=json_output_dir,
                    )
            except Exception as e:
                print(f"  ERROR: Configuration failed - {str(e)}")
                continue

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE!")
    print(f"All results saved to: {json_output_dir}")
    print(f"Total configurations processed: {config_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
