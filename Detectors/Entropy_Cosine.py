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
from datetime import datetime
import numpy as np

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

# ---------------------------------------------------------------------------
# CMS implementation with Numpy export
# ---------------------------------------------------------------------------
@dataclass
class CountMinSketch:
    width: int
    depth: int
    sketch: List[List[int]] = field(default_factory=list, init=False, repr=False)
    total_count: int = field(default=0, init=False)
    
    def __post_init__(self):
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
    
    def get_flattened_vector(self) -> np.ndarray:
        """Returns the 2D sketch as a flattened 1D numpy array."""
        return np.array([val for row in self.sketch for val in row], dtype=np.float64)

# ---------------------------------------------------------------------------
# Feature Tracker to hold 4 CMS arrays per epoch/window
# ---------------------------------------------------------------------------
@dataclass
class FlowSketcher:
    width: int = 1024
    depth: int = 5
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

def build_sketcher(flows: List[Flow], width: int = 1024, depth: int = 5) -> FlowSketcher:
    sketcher = FlowSketcher(width=width, depth=depth)
    for flow in flows:
        sketcher.add_flow(flow)
    return sketcher

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# First segment of flattened CMS (for reports / PNGs alongside cosine plots)
SKETCH_PREVIEW_LEN = 80


def _max_normalize_slice(vec: np.ndarray, k: int) -> np.ndarray:
    sl = np.asarray(vec[:k], dtype=np.float64)
    mx = float(sl.max()) if sl.size else 0.0
    if mx <= 0.0:
        return np.zeros_like(sl)
    return sl / mx


def _sketch_vecs_from_sketcher(sk: FlowSketcher) -> Dict[str, np.ndarray]:
    return {
        "src_ip": sk.src_ip_cms.get_flattened_vector(),
        "dst_ip": sk.dst_ip_cms.get_flattened_vector(),
        "src_port": sk.src_port_cms.get_flattened_vector(),
        "dst_port": sk.dst_port_cms.get_flattened_vector(),
    }


def _baseline_mean_raw_slices(
    baseline_epoch_sketchers: List[FlowSketcher], k: int
) -> Dict[str, np.ndarray]:
    if not baseline_epoch_sketchers:
        return {}
    feats = ["src_ip", "dst_ip", "src_port", "dst_port"]
    sums = {f: np.zeros(k, dtype=np.float64) for f in feats}
    for ep_sk in baseline_epoch_sketchers:
        ev = _sketch_vecs_from_sketcher(ep_sk)
        for f in feats:
            sums[f] += ev[f][:k]
    n = float(len(baseline_epoch_sketchers))
    return {f: sums[f] / n for f in feats}


def _build_sketch_preview(
    master_vecs: Dict[str, np.ndarray],
    obs_vecs: Dict[str, np.ndarray],
    epoch_z_scores: Dict[str, float],
    epoch_sims: Dict[str, float],
) -> Dict[str, object]:
    worst_feat = min(epoch_z_scores.keys(), key=lambda f: epoch_z_scores[f])
    k = SKETCH_PREVIEW_LEN
    m = _max_normalize_slice(master_vecs[worst_feat], k)
    v = _max_normalize_slice(obs_vecs[worst_feat], k)
    return {
        "worst_feature": worst_feat,
        "master": [round(float(x), 6) for x in m],
        "epoch": [round(float(x), 6) for x in v],
        "cosine": round(float(epoch_sims[worst_feat]), 6),
    }

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
# Core Configuration Logic
# ---------------------------------------------------------------------------
def run_sparse_configuration(directory: str, baseline_percentage: float, malicious_percentage: float,
                             dataset_name: str, output_dir: str,
                             attack_prob: float = 0.3, window_size: int = 10) -> dict:

    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))

    print(f"\n  [COSINE SIMILARITY] Config: Baseline={baseline_percentage}%, Malicious={malicious_percentage}%")
    
    baseline_indices = list(range(baseline_count))
    attack_pool = list(range(baseline_count, total_epochs))

    # Build windows
    windows = []
    for w_start in range(0, len(attack_pool), window_size):
        windows.append(attack_pool[w_start: w_start + window_size])

    attacked_epoch_set = set()
    for w_idx, window in enumerate(windows):
        if random.random() < attack_prob:
            for ep_idx in window:
                attacked_epoch_set.add(ep_idx)

    print(f"  Phase 1: Building Master Baseline CMS & Calculating Normal Variance...")
    
    # 1. Build the Master Sketch from ALL baseline flows
    master_sketcher = FlowSketcher()
    baseline_epoch_sketchers = []
    
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if not flows: continue
        
        # Build individual epoch sketch for variance calculation
        ep_sketcher = build_sketcher(flows)
        baseline_epoch_sketchers.append(ep_sketcher)
        
        # Add to master
        for flow in flows: master_sketcher.add_flow(flow)

    master_vecs = {
        'src_ip': master_sketcher.src_ip_cms.get_flattened_vector(),
        'dst_ip': master_sketcher.dst_ip_cms.get_flattened_vector(),
        'src_port': master_sketcher.src_port_cms.get_flattened_vector(),
        'dst_port': master_sketcher.dst_port_cms.get_flattened_vector()
    }

    # 2. Calculate the normal variance (mean and standard dev of similarity)
    baseline_sims = {'src_ip': [], 'dst_ip': [], 'src_port': [], 'dst_port': []}
    
    for ep_sketcher in baseline_epoch_sketchers:
        ep_vecs = {
            'src_ip': ep_sketcher.src_ip_cms.get_flattened_vector(),
            'dst_ip': ep_sketcher.dst_ip_cms.get_flattened_vector(),
            'src_port': ep_sketcher.src_port_cms.get_flattened_vector(),
            'dst_port': ep_sketcher.dst_port_cms.get_flattened_vector()
        }
        for feat in master_vecs:
            sim = calculate_cosine_similarity(master_vecs[feat], ep_vecs[feat])
            baseline_sims[feat].append(sim)

    baseline_stats = {}
    for feat in baseline_sims:
        baseline_stats[feat] = {
            'mean': float(np.mean(baseline_sims[feat])),
            'std': float(np.std(baseline_sims[feat])) if np.std(baseline_sims[feat]) > 0 else 1e-6
        }

    baseline_mean_raw = _baseline_mean_raw_slices(baseline_epoch_sketchers, SKETCH_PREVIEW_LEN)
    baseline_mean_sketch_preview = {
        f: [round(float(x), 6) for x in _max_normalize_slice(baseline_mean_raw[f], SKETCH_PREVIEW_LEN)]
        for f in baseline_mean_raw
    }

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "Cosine_Similarity",
            "sparse_mode": True,
            "sketch_preview_length": SKETCH_PREVIEW_LEN,
            "epoch_data_directory": os.path.abspath(directory),
            "baseline_mean_sketch_preview": baseline_mean_sketch_preview,
        },
        "baseline_stats": baseline_stats,
        "epochs": []
    }

    print(f"  Phase 2: Analyzing epochs with Cosine Distance Z-Scores...")
    
    # Threshold for flagging an anomaly: Z-score < -3.0 (3 standard deviations worse than normal)
    Z_SCORE_THRESHOLD = -3.0 
    
    for idx in attack_pool:
        epoch_name = os.path.splitext(os.path.basename(epoch_files[idx]))[0]
        clean_flows = parse_flows(epoch_files[idx])
        if not clean_flows: continue

        is_attacked = idx in attacked_epoch_set
        if is_attacked:
            num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
            combined_flows = clean_flows + generate_malicious_flows(num_malicious, clean_flows)
            observed_sketcher = build_sketcher(combined_flows)
        else:
            combined_flows = clean_flows
            observed_sketcher = build_sketcher(clean_flows)

        obs_vecs = {
            'src_ip': observed_sketcher.src_ip_cms.get_flattened_vector(),
            'dst_ip': observed_sketcher.dst_ip_cms.get_flattened_vector(),
            'src_port': observed_sketcher.src_port_cms.get_flattened_vector(),
            'dst_port': observed_sketcher.dst_port_cms.get_flattened_vector()
        }

        epoch_sims = {}
        epoch_z_scores = {}
        
        # Calculate similarity and Z-score for each feature
        for feat in master_vecs:
            sim = calculate_cosine_similarity(master_vecs[feat], obs_vecs[feat])
            epoch_sims[feat] = sim
            
            # Z-Score: (Observed - Mean) / StdDev
            z_score = (sim - baseline_stats[feat]['mean']) / baseline_stats[feat]['std']
            epoch_z_scores[feat] = z_score

        # We take the worst (most negative) Z-score among the 4 features as the overall anomaly score
        worst_z_score = min(epoch_z_scores.values())
        is_anomaly_predicted = bool(worst_z_score < Z_SCORE_THRESHOLD)

        epoch_result = {
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked_ground_truth": is_attacked,
            "ml_prediction_anomaly": is_anomaly_predicted,
            "ml_anomaly_score": round(float(worst_z_score), 6),  # Using this key so your plotting script works out of the box
            "total_flows_count": len(combined_flows),
            "similarities": {k: round(v, 6) for k, v in epoch_sims.items()},
            "z_scores": {k: round(v, 6) for k, v in epoch_z_scores.items()},
            "sketch_preview": _build_sketch_preview(master_vecs, obs_vecs, epoch_z_scores, epoch_sims),
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
    # Full attack mode - same logic, every epoch in phase 2 gets attacked.
    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        epoch_files = sorted([f for f in glob.glob(os.path.join(directory, '*')) if os.path.isfile(f)])

    total_epochs = len(epoch_files)
    baseline_count = max(1, int(total_epochs * baseline_percentage / 100))
    baseline_indices = list(range(baseline_count))
    attack_indices = list(range(baseline_count, total_epochs))

    print(f"  Phase 1: Building Master Baseline CMS & Calculating Normal Variance...")
    master_sketcher = FlowSketcher()
    baseline_epoch_sketchers = []
    
    for idx in baseline_indices:
        flows = parse_flows(epoch_files[idx])
        if not flows: continue
        baseline_epoch_sketchers.append(build_sketcher(flows))
        for flow in flows: master_sketcher.add_flow(flow)

    master_vecs = {
        'src_ip': master_sketcher.src_ip_cms.get_flattened_vector(),
        'dst_ip': master_sketcher.dst_ip_cms.get_flattened_vector(),
        'src_port': master_sketcher.src_port_cms.get_flattened_vector(),
        'dst_port': master_sketcher.dst_port_cms.get_flattened_vector()
    }

    baseline_sims = {'src_ip': [], 'dst_ip': [], 'src_port': [], 'dst_port': []}
    for ep_sketcher in baseline_epoch_sketchers:
        ep_vecs = {
            'src_ip': ep_sketcher.src_ip_cms.get_flattened_vector(),
            'dst_ip': ep_sketcher.dst_ip_cms.get_flattened_vector(),
            'src_port': ep_sketcher.src_port_cms.get_flattened_vector(),
            'dst_port': ep_sketcher.dst_port_cms.get_flattened_vector()
        }
        for feat in master_vecs:
            baseline_sims[feat].append(calculate_cosine_similarity(master_vecs[feat], ep_vecs[feat]))

    baseline_stats = {feat: {'mean': float(np.mean(baseline_sims[feat])), 'std': float(np.std(baseline_sims[feat])) if np.std(baseline_sims[feat]) > 0 else 1e-6} for feat in baseline_sims}

    baseline_mean_raw = _baseline_mean_raw_slices(baseline_epoch_sketchers, SKETCH_PREVIEW_LEN)
    baseline_mean_sketch_preview = {
        f: [round(float(x), 6) for x in _max_normalize_slice(baseline_mean_raw[f], SKETCH_PREVIEW_LEN)]
        for f in baseline_mean_raw
    }

    results = {
        "metadata": {
            "dataset_name": dataset_name,
            "sparse_mode": False,
            "method": "Cosine_Similarity",
            "sketch_preview_length": SKETCH_PREVIEW_LEN,
            "epoch_data_directory": os.path.abspath(directory),
            "baseline_mean_sketch_preview": baseline_mean_sketch_preview,
        },
        "baseline_stats": baseline_stats,
        "epochs": []
    }

    print(f"  Phase 2: Analyzing attack epochs with Cosine Distance Z-Scores...")
    Z_SCORE_THRESHOLD = -3.0

    for idx in attack_indices:
        epoch_name = os.path.splitext(os.path.basename(epoch_files[idx]))[0]
        clean_flows = parse_flows(epoch_files[idx])
        if not clean_flows: continue

        num_malicious = int(len(clean_flows) * (malicious_percentage / 100))
        combined_flows = clean_flows + generate_malicious_flows(num_malicious, clean_flows)
        observed_sketcher = build_sketcher(combined_flows)

        obs_vecs = {
            'src_ip': observed_sketcher.src_ip_cms.get_flattened_vector(),
            'dst_ip': observed_sketcher.dst_ip_cms.get_flattened_vector(),
            'src_port': observed_sketcher.src_port_cms.get_flattened_vector(),
            'dst_port': observed_sketcher.dst_port_cms.get_flattened_vector()
        }

        epoch_sims = {feat: calculate_cosine_similarity(master_vecs[feat], obs_vecs[feat]) for feat in master_vecs}
        epoch_z_scores = {feat: (epoch_sims[feat] - baseline_stats[feat]['mean']) / baseline_stats[feat]['std'] for feat in master_vecs}
        
        worst_z_score = min(epoch_z_scores.values())

        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": epoch_name,
            "attacked_ground_truth": True,
            "ml_prediction_anomaly": bool(worst_z_score < Z_SCORE_THRESHOLD),
            "ml_anomaly_score": round(float(worst_z_score), 6),
            "total_flows_count": len(combined_flows),
            "similarities": {k: round(v, 6) for k, v in epoch_sims.items()},
            "z_scores": {k: round(v, 6) for k, v in epoch_z_scores.items()},
            "sketch_preview": _build_sketch_preview(master_vecs, obs_vecs, epoch_z_scores, epoch_sims),
        })

    output_filename = f"{dataset_name}_{baseline_percentage}_{malicious_percentage}_ml.json"
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CMS Cosine Similarity Analysis")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--attack-prob", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=10)
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