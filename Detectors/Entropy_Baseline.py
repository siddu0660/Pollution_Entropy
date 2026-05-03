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
        if cardinality == 0:
            return 0.0
        return math.pow(2, entropy) / cardinality
    
    def get_src_ip_count(self, src_ip: str) -> int:
        return self.src_ip_cms.estimate(src_ip)
    
    def get_dst_ip_count(self, dst_ip: str) -> int:
        return self.dst_ip_cms.estimate(dst_ip)
    
    def get_src_port_count(self, src_port: str) -> int:
        return self.src_port_cms.estimate(src_port)
    
    def get_dst_port_count(self, dst_port: str) -> int:
        return self.dst_port_cms.estimate(dst_port)
    
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
    # Combined-pair entropies
    src_dst_ip_entropy: float
    src_dst_ip_cardinality: float
    src_dst_port_entropy: float
    src_dst_port_cardinality: float

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
    }


def calculate_baseline_entropies(baseline_metrics: List[EntropyMetrics]) -> Dict[str, float]:
    if not baseline_metrics:
        return {}
    
    baseline = {
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
    }
    return baseline

def calculate_entropy_changes(baseline: Dict[str, float], current: EntropyMetrics) -> Dict[str, float]:
    changes = {
        "total_flows_change": current.total_flows - baseline["total_flows"],
        "unique_src_ips_change": current.unique_src_ips - baseline["unique_src_ips"],
        "unique_dst_ips_change": current.unique_dst_ips - baseline["unique_dst_ips"],
        "unique_src_ports_change": current.unique_src_ports - baseline["unique_src_ports"],
        "unique_dst_ports_change": current.unique_dst_ports - baseline["unique_dst_ports"],
        "src_ip_entropy_change": current.src_ip_entropy - baseline["src_ip_entropy"],
        "src_ip_cardinality_change": current.src_ip_cardinality - baseline["src_ip_cardinality"],
        "dst_ip_entropy_change": current.dst_ip_entropy - baseline["dst_ip_entropy"],
        "dst_ip_cardinality_change": current.dst_ip_cardinality - baseline["dst_ip_cardinality"],
        "src_port_entropy_change": current.src_port_entropy - baseline["src_port_entropy"],
        "src_port_cardinality_change": current.src_port_cardinality - baseline["src_port_cardinality"],
        "dst_port_entropy_change": current.dst_port_entropy - baseline["dst_port_entropy"],
        "dst_port_cardinality_change": current.dst_port_cardinality - baseline["dst_port_cardinality"],
        "src_ip_uniformity_change": current.src_ip_uniformity - baseline["src_ip_uniformity"],
        "dst_ip_uniformity_change": current.dst_ip_uniformity - baseline["dst_ip_uniformity"],
        "src_dst_ratio_change": current.src_dst_ratio - baseline["src_dst_ratio"],
        "src_dst_ip_entropy_change": current.src_dst_ip_entropy - baseline["src_dst_ip_entropy"],
        "src_dst_ip_cardinality_change": current.src_dst_ip_cardinality - baseline["src_dst_ip_cardinality"],
        "src_dst_port_entropy_change": current.src_dst_port_entropy - baseline["src_dst_port_entropy"],
        "src_dst_port_cardinality_change": current.src_dst_port_cardinality - baseline["src_dst_port_cardinality"],
    }
    return changes

def run_single_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  Configuration: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%")
    print(f"  Total epochs: {total_epochs}, Baseline epochs: {baseline_count}")

    baseline_indices = list(range(baseline_count))
    attack_indices = list(range(baseline_count, total_epochs))

    print(f"  Phase 1: Calculating baseline entropies...")
    baseline_metrics_list = []

    for idx in baseline_indices:
        epoch_file = epoch_files[idx]
        flows = parse_flows(epoch_file)
        if not flows:
            continue
        metrics = calculate_all_entropies(flows)
        baseline_metrics_list.append(metrics)

    baseline = calculate_baseline_entropies(baseline_metrics_list)
    print(f"  Baseline established from {len(baseline_metrics_list)} epochs.")

    print(f"  Phase 2: Analyzing attack epochs...")
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
        "epochs": []
    }

    for idx in attack_indices:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]

        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        clean_metrics = calculate_all_entropies(clean_flows)

        num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
        malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
        combined_flows = clean_flows + malicious_flows

        malicious_metrics = calculate_all_entropies(combined_flows)
        changes = calculate_entropy_changes(baseline, malicious_metrics)

        epoch_result = {
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked": True,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "clean_metrics": metrics_to_dict(clean_metrics),
            "malicious_metrics": metrics_to_dict(malicious_metrics),
            "entropy_changes": {k: round(v, 6) if isinstance(v, float) else v for k, v in changes.items()}
        }

        results["epochs"].append(epoch_result)

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")

    return results


def run_sparse_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str,
                             attack_prob: float = 0.3, window_size: int = 10) -> dict:
    """Sparse attack mode: attack epochs are grouped into windows of `window_size`.
    Each window is independently selected for attack with probability `attack_prob`.
    Only epochs inside a selected window receive malicious traffic injection.
    """

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  [SPARSE] Configuration: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%,"
          f" AttackProb={attack_prob}, WindowSize={window_size}")
    print(f"  Total epochs: {total_epochs}, Baseline epochs: {baseline_count}")

    baseline_indices = list(range(baseline_count))
    attack_pool = list(range(baseline_count, total_epochs))

    # Build windows of `window_size` epochs from the attack pool
    windows = []
    for w_start in range(0, len(attack_pool), window_size):
        window_epoch_indices = attack_pool[w_start: w_start + window_size]
        windows.append(window_epoch_indices)

    # Randomly select which windows to attack
    attacked_windows = []
    clean_windows = []
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
            "window_index": w_idx,
            "epoch_indices": windows[w_idx],
            "epoch_range": [windows[w_idx][0], windows[w_idx][-1]] if windows[w_idx] else []
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

    # Phase 1: baseline
    print(f"  Phase 1: Calculating baseline entropies...")
    baseline_metrics_list = []
    for idx in baseline_indices:
        epoch_file = epoch_files[idx]
        flows = parse_flows(epoch_file)
        if not flows:
            continue
        metrics = calculate_all_entropies(flows)
        baseline_metrics_list.append(metrics)

    baseline = calculate_baseline_entropies(baseline_metrics_list)
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
        "epochs": []
    }

    # Phase 2: process all attack-pool epochs
    print(f"  Phase 2: Analyzing epochs (sparse attack)...")
    for idx in attack_pool:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]

        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        is_attacked = idx in attacked_epoch_set
        clean_metrics = calculate_all_entropies(clean_flows)

        if is_attacked:
            num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
            malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
            combined_flows = clean_flows + malicious_flows
            observed_metrics = calculate_all_entropies(combined_flows)
        else:
            num_malicious = 0
            combined_flows = clean_flows
            observed_metrics = clean_metrics

        changes = calculate_entropy_changes(baseline, observed_metrics)

        epoch_result = {
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked": is_attacked,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "clean_metrics": metrics_to_dict(clean_metrics),
            "malicious_metrics": metrics_to_dict(observed_metrics),
            "entropy_changes": {k: round(v, 6) if isinstance(v, float) else v for k, v in changes.items()}
        }

        results["epochs"].append(epoch_result)

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_sparse.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")

    return results


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Entropy Analysis System - Multi-Configuration"
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Directory containing epoch flow files")
    parser.add_argument("--sparse", action="store_true",
                        help="Sparse attack mode: attack only randomly selected windows of epochs")
    parser.add_argument("--attack-prob", type=float, default=0.3,
                        help="Probability of selecting each window for attack in sparse mode (default: 0.3)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Number of epochs per attack window in sparse mode (default: 10)")
    args = parser.parse_args()

    print("="*80)
    mode_label = "SPARSE" if args.sparse else "FULL"
    print(f"BASELINE ENTROPY ANALYSIS SYSTEM - MULTI-CONFIGURATION [{mode_label} MODE]")
    print("="*80)

    if args.directory:
        directory = args.directory
    else:
        directory = input("\nEnter the directory path containing epoch flow files: ").strip()

    dataset_name = os.path.basename(os.path.normpath(directory))

    json_output_dir = os.path.join(directory, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    print(f"\nOutput directory created: {json_output_dir}")

    BASELINE_PERCENTAGES = [20.0, 30.0]
    MALICIOUS_PERCENTAGES = [2.0]

    print(f"\nDataset: {dataset_name}")
    if args.sparse:
        print(f"Mode: SPARSE  |  Attack probability: {args.attack_prob}  |  Window size: {args.window_size} epochs")
    print(f"Baseline percentages to test: {BASELINE_PERCENTAGES}")
    print(f"Malicious percentages to test: {MALICIOUS_PERCENTAGES}")
    print(f"Total configurations: {len(BASELINE_PERCENTAGES) * len(MALICIOUS_PERCENTAGES)}")
    print("-" * 80)

    config_count = 0
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
