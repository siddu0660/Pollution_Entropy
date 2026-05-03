from typing import Tuple, Dict, List
from dataclasses import dataclass, field
import zlib
import hashlib
import math
import os
import random
import glob
import json
import statistics
from collections import defaultdict
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
# Count-Min Sketch (used for memory-efficient frequency estimation)
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
        self.sketch = [[0] * self.width for _ in range(self.depth)]

    def hash(self, item: str, seed: int) -> int:
        hash_value = zlib.crc32(f"{item}_{seed}".encode('utf-8'))
        return hash_value % self.width

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            self.sketch[i][self.hash(item, i)] += count
        self.total_count += count

    def estimate(self, item: str) -> int:
        return int(min(self.sketch[i][self.hash(item, i)] for i in range(self.depth)))


# ---------------------------------------------------------------------------
# FlowTracker: builds per-feature CMS + tracks unique keys
# ---------------------------------------------------------------------------
@dataclass
class FlowTracker:
    width: int = 1024
    depth: int = 5
    src_ip_cms: CountMinSketch = field(init=False)
    dst_ip_cms: CountMinSketch = field(init=False)
    src_port_cms: CountMinSketch = field(init=False)
    dst_port_cms: CountMinSketch = field(init=False)
    src_ip_keys: set = field(default_factory=set, init=False)
    dst_ip_keys: set = field(default_factory=set, init=False)
    src_port_keys: set = field(default_factory=set, init=False)
    dst_port_keys: set = field(default_factory=set, init=False)

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
        self.src_ip_keys.add(flow.src_ip)
        self.dst_ip_keys.add(flow.dst_ip)
        self.src_port_keys.add(flow.src_port)
        self.dst_port_keys.add(flow.dst_port)

    def get_prob_dist(self, cms: CountMinSketch, keys: set) -> Dict[str, float]:
        """Return a probability distribution dict estimated via CMS."""
        if cms.total_count == 0:
            return {}
        dist = {}
        for key in keys:
            freq = cms.estimate(key)
            if freq > 0:
                dist[key] = freq / cms.total_count
        # Normalise (CMS can over-count; this keeps sum=1)
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    def src_ip_dist(self) -> Dict[str, float]:
        return self.get_prob_dist(self.src_ip_cms, self.src_ip_keys)

    def dst_ip_dist(self) -> Dict[str, float]:
        return self.get_prob_dist(self.dst_ip_cms, self.dst_ip_keys)

    def src_port_dist(self) -> Dict[str, float]:
        return self.get_prob_dist(self.src_port_cms, self.src_port_keys)

    def dst_port_dist(self) -> Dict[str, float]:
        return self.get_prob_dist(self.dst_port_cms, self.dst_port_keys)


def build_tracker(flows: List[Flow], width: int = 1024, depth: int = 5) -> FlowTracker:
    tracker = FlowTracker(width=width, depth=depth)
    for flow in flows:
        tracker.add_flow(flow)
    return tracker


# ---------------------------------------------------------------------------
# Divergence functions
# ---------------------------------------------------------------------------

def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """KL divergence D_KL(P || Q).  Returns ∞ if Q has zero mass where P > 0.
    Uses additive smoothing (epsilon) on Q to avoid division by zero.
    All keys in P are iterated; missing Q keys are treated as epsilon.
    Result is in nats (natural log).
    """
    kl = 0.0
    for key, p_val in p.items():
        if p_val <= 0:
            continue
        q_val = q.get(key, 0.0) + epsilon
        kl += p_val * math.log(p_val / q_val)
    return kl


def js_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """Jensen-Shannon divergence JSD(P || Q).
    Symmetric, bounded in [0, ln(2)] (nats).  Returns a value in [0, 1]
    by dividing by ln(2) so the result is in bits-equivalent [0, 1].
    M = 0.5*(P + Q) over the union of keys.
    """
    all_keys = set(p.keys()) | set(q.keys())
    m: Dict[str, float] = {}
    for key in all_keys:
        m[key] = 0.5 * (p.get(key, 0.0) + q.get(key, 0.0))

    jsd = 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)
    # Normalise to [0, 1]
    return jsd / math.log(2)


# ---------------------------------------------------------------------------
# Baseline distribution: aggregate all baseline flows into one distribution
# ---------------------------------------------------------------------------

def build_baseline_distributions(baseline_flows_list: List[List[Flow]],
                                  width: int = 1024, depth: int = 5
                                  ) -> Dict[str, Dict[str, float]]:
    """Aggregate all baseline flows into a single reference distribution
    per feature dimension."""
    all_flows: List[Flow] = []
    for flows in baseline_flows_list:
        all_flows.extend(flows)

    if not all_flows:
        return {"src_ip": {}, "dst_ip": {}, "src_port": {}, "dst_port": {}}

    tracker = build_tracker(all_flows, width=width, depth=depth)
    return {
        "src_ip": tracker.src_ip_dist(),
        "dst_ip": tracker.dst_ip_dist(),
        "src_port": tracker.src_port_dist(),
        "dst_port": tracker.dst_port_dist(),
        "total_flows": len(all_flows),
        "unique_src_ips": len(tracker.src_ip_keys),
        "unique_dst_ips": len(tracker.dst_ip_keys),
        "unique_src_ports": len(tracker.src_port_keys),
        "unique_dst_ports": len(tracker.dst_port_keys),
    }


# ---------------------------------------------------------------------------
# Per-epoch divergence metrics
# ---------------------------------------------------------------------------

@dataclass
class DivergenceMetrics:
    total_flows: int
    unique_src_ips: int
    unique_dst_ips: int
    unique_src_ports: int
    unique_dst_ports: int
    # JS divergence against baseline (symmetric, [0,1])
    src_ip_js: float
    dst_ip_js: float
    src_port_js: float
    dst_port_js: float
    # KL divergence epoch || baseline (asymmetric)
    src_ip_kl: float
    dst_ip_kl: float
    src_port_kl: float
    dst_port_kl: float
    # Combined JS score (mean of four features)
    combined_js: float


def calculate_divergence_metrics(flows: List[Flow],
                                  baseline_dists: Dict[str, Dict[str, float]],
                                  width: int = 1024, depth: int = 5) -> DivergenceMetrics:
    """Compute JS and KL divergence between `flows` distribution and `baseline_dists`."""
    tracker = build_tracker(flows, width=width, depth=depth)

    epoch_dists = {
        "src_ip": tracker.src_ip_dist(),
        "dst_ip": tracker.dst_ip_dist(),
        "src_port": tracker.src_port_dist(),
        "dst_port": tracker.dst_port_dist(),
    }

    features = ["src_ip", "dst_ip", "src_port", "dst_port"]
    js_vals = {}
    kl_vals = {}
    for feat in features:
        ep_dist = epoch_dists[feat]
        bl_dist = baseline_dists.get(feat, {})
        js_vals[feat] = js_divergence(ep_dist, bl_dist)
        kl_vals[feat] = kl_divergence(ep_dist, bl_dist)

    combined = statistics.mean(js_vals[f] for f in features)

    return DivergenceMetrics(
        total_flows=len(flows),
        unique_src_ips=len(tracker.src_ip_keys),
        unique_dst_ips=len(tracker.dst_ip_keys),
        unique_src_ports=len(tracker.src_port_keys),
        unique_dst_ports=len(tracker.dst_port_keys),
        src_ip_js=js_vals["src_ip"],
        dst_ip_js=js_vals["dst_ip"],
        src_port_js=js_vals["src_port"],
        dst_port_js=js_vals["dst_port"],
        src_ip_kl=kl_vals["src_ip"],
        dst_ip_kl=kl_vals["dst_ip"],
        src_port_kl=kl_vals["src_port"],
        dst_port_kl=kl_vals["dst_port"],
        combined_js=combined,
    )


def metrics_to_dict(m: DivergenceMetrics) -> dict:
    return {
        "total_flows": m.total_flows,
        "unique_src_ips": m.unique_src_ips,
        "unique_dst_ips": m.unique_dst_ips,
        "unique_src_ports": m.unique_src_ports,
        "unique_dst_ports": m.unique_dst_ports,
        "src_ip_js": round(m.src_ip_js, 8),
        "dst_ip_js": round(m.dst_ip_js, 8),
        "src_port_js": round(m.src_port_js, 8),
        "dst_port_js": round(m.dst_port_js, 8),
        "src_ip_kl": round(m.src_ip_kl, 8),
        "dst_ip_kl": round(m.dst_ip_kl, 8),
        "src_port_kl": round(m.src_port_kl, 8),
        "dst_port_kl": round(m.dst_port_kl, 8),
        "combined_js": round(m.combined_js, 8),
    }


def divergence_vs_baseline(polluted_m: DivergenceMetrics) -> dict:
    """Absolute JS/KL divergence of the epoch vs the baseline distribution."""
    return {
        "src_ip_js": round(polluted_m.src_ip_js, 8),
        "dst_ip_js": round(polluted_m.dst_ip_js, 8),
        "src_port_js": round(polluted_m.src_port_js, 8),
        "dst_port_js": round(polluted_m.dst_port_js, 8),
        "src_ip_kl": round(polluted_m.src_ip_kl, 8),
        "dst_ip_kl": round(polluted_m.dst_ip_kl, 8),
        "src_port_kl": round(polluted_m.src_port_kl, 8),
        "dst_port_kl": round(polluted_m.dst_port_kl, 8),
        "combined_js": round(polluted_m.combined_js, 8),
        "total_flows": polluted_m.total_flows,
    }


# ---------------------------------------------------------------------------
# Malicious flow generation
# ---------------------------------------------------------------------------

def generate_malicious_flows(num_flows: int, base_flows: List[Flow]) -> List[Flow]:
    flows = []
    for _ in range(num_flows):
        src_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        dst_ip = f"{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}{random.randint(0,255):03d}"
        src_port = f"{random.randint(0, 65535):05d}"
        dst_port_val = random.choice([80, 443, 808, 22, 53, 330, 543, random.randint(0, 65535)])
        dst_port = f"{dst_port_val % 65535:05d}"
        protocol = f"{random.choice([6, 17]):02d}"
        flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    return flows


# ---------------------------------------------------------------------------
# Baseline summary for JSON output (scalar stats, not raw dist)
# ---------------------------------------------------------------------------

def baseline_summary(baseline_dists: Dict) -> dict:
    return {
        "total_flows": baseline_dists.get("total_flows", 0),
        "unique_src_ips": baseline_dists.get("unique_src_ips", 0),
        "unique_dst_ips": baseline_dists.get("unique_dst_ips", 0),
        "unique_src_ports": baseline_dists.get("unique_src_ports", 0),
        "unique_dst_ports": baseline_dists.get("unique_dst_ports", 0),
        "src_ip_vocab_size": len(baseline_dists.get("src_ip", {})),
        "dst_ip_vocab_size": len(baseline_dists.get("dst_ip", {})),
        "src_port_vocab_size": len(baseline_dists.get("src_port", {})),
        "dst_port_vocab_size": len(baseline_dists.get("dst_port", {})),
    }


# ---------------------------------------------------------------------------
# Run single (full-attack) configuration
# ---------------------------------------------------------------------------

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

    # Phase 1: build baseline distributions from all baseline epochs
    print(f"  Phase 1: Building baseline distributions...")
    baseline_flows_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_flows_list.append(flows)

    baseline_dists = build_baseline_distributions(baseline_flows_list)
    print(f"  Baseline built from {len(baseline_flows_list)} epochs "
          f"({baseline_dists.get('total_flows', 0):,} total flows).")

    # Phase 2: measure JS/KL divergence for each attack epoch
    print(f"  Phase 2: Analyzing attack epochs...")
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "directory": directory,
            "divergence_method": "Jensen-Shannon + KL",
            "total_epochs": total_epochs,
            "baseline_epoch_count": baseline_count,
            "baseline_percentage": baseline_percentage,
            "malicious_percentage": malicious_percentage,
            "baseline_indices": baseline_indices,
            "attack_indices": attack_indices,
            "sparse_mode": False,
        },
        "baseline": baseline_summary(baseline_dists),
        "epochs": []
    }

    for idx in attack_indices:
        epoch_file = epoch_files[idx]
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]

        clean_flows = parse_flows(epoch_file)
        if not clean_flows:
            continue

        num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
        malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
        combined_flows = clean_flows + malicious_flows

        polluted_metrics = calculate_divergence_metrics(combined_flows, baseline_dists)

        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked": True,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "polluted_metrics": metrics_to_dict(polluted_metrics),
            "divergence_vs_baseline": divergence_vs_baseline(polluted_metrics),
        })

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")
    return results


# ---------------------------------------------------------------------------
# Run sparse configuration
# ---------------------------------------------------------------------------

def run_sparse_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                              dataset_name: str, output_dir: str,
                              attack_prob: float = 0.3, window_size: int = 10) -> dict:
    """Sparse attack mode: attack epochs are grouped into windows of `window_size`.
    Each window is independently selected for attack with probability `attack_prob`.
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

    attacked_epoch_set = {ep for w_idx in attacked_windows for ep in windows[w_idx]}

    attacked_window_info = [
        {
            "window_index": w_idx,
            "epoch_indices": windows[w_idx],
            "epoch_range": [windows[w_idx][0], windows[w_idx][-1]] if windows[w_idx] else [],
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

    # Phase 1: build baseline distributions
    print(f"  Phase 1: Building baseline distributions...")
    baseline_flows_list = []
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if flows:
            baseline_flows_list.append(flows)

    baseline_dists = build_baseline_distributions(baseline_flows_list)
    print(f"  Baseline built from {len(baseline_flows_list)} epochs "
          f"({baseline_dists.get('total_flows', 0):,} total flows).")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "directory": directory,
            "divergence_method": "Jensen-Shannon + KL",
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
        "baseline": baseline_summary(baseline_dists),
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

        if is_attacked:
            num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
            malicious_flows = generate_malicious_flows(num_malicious, clean_flows)
            combined_flows = clean_flows + malicious_flows
            observed_metrics = calculate_divergence_metrics(combined_flows, baseline_dists)
        else:
            num_malicious = 0
            combined_flows = clean_flows
            observed_metrics = calculate_divergence_metrics(clean_flows, baseline_dists)

        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked": is_attacked,
            "clean_flows_count": len(clean_flows),
            "malicious_flows_count": num_malicious,
            "total_flows_count": len(combined_flows),
            "polluted_metrics": metrics_to_dict(observed_metrics),
            "divergence_vs_baseline": divergence_vs_baseline(observed_metrics),
        })

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_sparse.json"
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {output_filename}")
    print(f"  Total epochs analyzed: {len(results['epochs'])}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="JS/KL Divergence Analysis System - Multi-Configuration"
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

    print("=" * 80)
    mode_label = "SPARSE" if args.sparse else "FULL"
    print(f"JS/KL DIVERGENCE ANALYSIS SYSTEM - MULTI-CONFIGURATION [{mode_label} MODE]")
    print("=" * 80)

    if args.directory:
        directory = args.directory
    else:
        directory = input("\nEnter the directory path containing epoch flow files: ").strip()

    dataset_name = os.path.basename(os.path.normpath(directory))

    json_output_dir = os.path.join(directory, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    print(f"\nOutput directory created: {json_output_dir}")

    BASELINE_PERCENTAGES = [5.0, 10.0]
    MALICIOUS_PERCENTAGES = [0.0, 1.0, 2.0, 5.0]

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
