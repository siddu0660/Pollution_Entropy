"""
Entropy_CUSUM.py  –  CUSUM Sequential Change-Point Detector
=============================================================

Key Insight
-----------
The previous scripts treat every epoch independently.  They compare each epoch
to a static baseline but ignore the *temporal order* of epochs.  A CUSUM
(Cumulative Sum Control Chart) detector instead accumulates per-epoch anomaly
scores over time and raises an alarm when the cumulative sum crosses a
threshold — it is statistically optimal for detecting persistent but small
shifts, exactly the scenario where low-percentage attacks fail to stand out
in a single epoch.

Design
------
1. Use **any** per-epoch scalar signal as input.  This script uses the
   combined_js score from a lightweight JS divergence (same idea as
   Entropy_JS.py) but also tracks three secondary signals:
     - new_src_ip_ratio  (fraction of flows from IPs not in baseline)
     - topk_src_fraction (fraction from top-5 src IPs — traffic concentration)
     - flow_volume_ratio (epoch-flow-count / baseline-mean-flow-count)

2. For each signal compute CUSUM statistics:
     C_plus[t]  = max(0, C_plus[t-1]  + (x[t] - μ₀ - k))
     C_minus[t] = max(0, C_minus[t-1] + (μ₀ + k - x[t]))
   where μ₀ is the baseline mean, k = 0.5 σ₀ (the "allowance" / slack).

3. An alarm fires when C_plus[t] > h (default h = 4 σ₀), meaning a positive
   shift of the anomaly signal has accumulated to ≥ 4 standard deviations.

4. Output includes the raw signal value, C_plus, C_minus, and a boolean
   `cusum_alarm` per epoch — perfect for downstream plotting and evaluation.

Usage
-----
  python Entropy_CUSUM.py <directory> [--sparse] [--attack-prob 0.3] \
                            [--window-size 10] [--h-factor 4.0] [--k-factor 0.5]
"""

from typing import List, Dict, Set
from dataclasses import dataclass, field
import zlib
import math
import os
import random
import glob
import json
import statistics
from collections import defaultdict
from datetime import datetime


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
            flows.append(Flow(line[0:12], line[12:24], line[24:29],
                              line[29:34], line[34:36]))
    return flows


# ---------------------------------------------------------------------------
# Lightweight CMS + JS divergence (copied from Entropy_JS.py for self-containment)
# ---------------------------------------------------------------------------

class CMS:
    def __init__(self, width=512, depth=4):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]
        self.total = 0

    def _h(self, item, row):
        return zlib.crc32(f"{item}_{row}".encode()) % self.w

    def add(self, item):
        for r in range(self.d):
            self.table[r][self._h(item, r)] += 1
        self.total += 1

    def est(self, item):
        return min(self.table[r][self._h(item, r)] for r in range(self.d))


def _prob_dist(cms: CMS, keys: Set[str]) -> Dict[str, float]:
    if cms.total == 0:
        return {}
    raw = {k: cms.est(k) for k in keys if cms.est(k) > 0}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()} if tot else {}


def _js(p: Dict[str, float], q: Dict[str, float], eps=1e-10) -> float:
    all_keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in all_keys}

    def kl(a, b):
        return sum(av * math.log(av / (b.get(k, 0) + eps))
                   for k, av in a.items() if av > 0)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def build_baseline_dists(baseline_flows_list):
    all_flows = [f for epoch in baseline_flows_list for f in epoch]
    if not all_flows:
        return {}

    cms_src = CMS(); cms_dst = CMS(); cms_sp = CMS(); cms_dp = CMS()
    src_keys = set(); dst_keys = set(); sp_keys = set(); dp_keys = set()

    for flow in all_flows:
        cms_src.add(flow.src_ip);  src_keys.add(flow.src_ip)
        cms_dst.add(flow.dst_ip);  dst_keys.add(flow.dst_ip)
        cms_sp.add(flow.src_port); sp_keys.add(flow.src_port)
        cms_dp.add(flow.dst_port); dp_keys.add(flow.dst_port)

    return {
        "src_ip":   (_prob_dist(cms_src, src_keys), src_keys),
        "dst_ip":   (_prob_dist(cms_dst, dst_keys), dst_keys),
        "src_port": (_prob_dist(cms_sp,  sp_keys),  sp_keys),
        "dst_port": (_prob_dist(cms_dp,  dp_keys),  dp_keys),
        "all_src_ips": src_keys,
    }


def epoch_signal(flows: List[Flow], bl_dists: dict, bl_src_set: Set[str]) -> Dict[str, float]:
    if not flows:
        return {"js": 0.0, "new_src_ratio": 0.0, "topk_src": 0.0}

    cms_src = CMS(); cms_dst = CMS(); cms_sp = CMS(); cms_dp = CMS()
    src_keys = set()

    for flow in flows:
        cms_src.add(flow.src_ip);  src_keys.add(flow.src_ip)
        cms_dst.add(flow.dst_ip)
        cms_sp.add(flow.src_port)
        cms_dp.add(flow.dst_port)

    ep_dists = {
        "src_ip":   _prob_dist(cms_src, src_keys),
        "dst_ip":   _prob_dist(cms_dst, set(f.dst_ip for f in flows)),
        "src_port": _prob_dist(cms_sp,  set(f.src_port for f in flows)),
        "dst_port": _prob_dist(cms_dp,  set(f.dst_port for f in flows)),
    }

    js_vals = []
    for feat in ["src_ip", "dst_ip", "src_port", "dst_port"]:
        bl_d, _ = bl_dists[feat]
        js_vals.append(_js(ep_dists[feat], bl_d))
    combined_js = statistics.mean(js_vals)

    # Fraction of flows from IPs NOT in baseline
    new_src = sum(1 for f in flows if f.src_ip not in bl_src_set)
    new_src_ratio = new_src / len(flows)

    # Top-5 src concentration
    src_counts = defaultdict(int)
    for f in flows:
        src_counts[f.src_ip] += 1
    top5 = sorted(src_counts.values(), reverse=True)[:5]
    topk_src = sum(top5) / len(flows)

    return {
        "js":            round(combined_js, 8),
        "new_src_ratio": round(new_src_ratio, 8),
        "topk_src":      round(topk_src, 8),
    }


# ---------------------------------------------------------------------------
# CUSUM state per signal
# ---------------------------------------------------------------------------

@dataclass
class CUSUMState:
    mean: float          # baseline mean μ₀
    std: float           # baseline std σ₀
    k_factor: float      # slack coefficient (k = k_factor * σ₀)
    h_factor: float      # alarm threshold (h = h_factor * σ₀)
    c_plus: float = 0.0
    c_minus: float = 0.0

    @property
    def k(self):
        return self.k_factor * max(self.std, 1e-12)

    @property
    def h(self):
        return self.h_factor * max(self.std, 1e-12)

    def update(self, x: float) -> bool:
        """Update CUSUM and return True if upward alarm fires."""
        self.c_plus  = max(0.0, self.c_plus  + (x - self.mean - self.k))
        self.c_minus = max(0.0, self.c_minus + (self.mean - self.k - x))
        return self.c_plus > self.h  or self.c_minus > self.h

    def snapshot(self):
        return {"c_plus": round(self.c_plus, 6),
                "c_minus": round(self.c_minus, 6),
                "threshold_h": round(self.h, 6)}


def build_cusum_states(baseline_signals: List[Dict[str, float]],
                        k_factor: float, h_factor: float) -> Dict[str, CUSUMState]:
    keys = ["js", "new_src_ratio", "topk_src"]
    states = {}
    for key in keys:
        vals = [s[key] for s in baseline_signals if s]
        m = statistics.mean(vals)   if vals else 0.0
        s = statistics.stdev(vals)  if len(vals) > 1 else 0.0
        states[key] = CUSUMState(mean=m, std=s, k_factor=k_factor, h_factor=h_factor)
    return states


# ---------------------------------------------------------------------------
# Malicious flow generation
# ---------------------------------------------------------------------------

def generate_malicious_flows(n: int, base_flows: List[Flow]) -> List[Flow]:
    out = []
    for _ in range(n):
        src = (f"{random.randint(0,255):03d}{random.randint(0,255):03d}"
               f"{random.randint(0,255):03d}{random.randint(0,255):03d}")
        dst = (f"{random.randint(0,255):03d}{random.randint(0,255):03d}"
               f"{random.randint(0,255):03d}{random.randint(0,255):03d}")
        sp  = f"{random.randint(0,65535):05d}"
        dp  = f"{random.choice([80,443,808,22,53,330,543,random.randint(0,65535)]) % 65535:05d}"
        pr  = f"{random.choice([6,17]):02d}"
        out.append(Flow(src, dst, sp, dp, pr))
    return out


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------

def _epoch_files(directory: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not files:
        files = sorted([f for f in glob.glob(os.path.join(directory, '*'))
                        if os.path.isfile(f)])
    return files


def _run_epochs(epoch_files, epoch_indices, attacked_set,
                malicious_pct, bl_dists, bl_src_set, cusum_states):
    epoch_results = []
    for idx in epoch_indices:
        clean = parse_flows(epoch_files[idx])
        if not clean:
            continue

        is_attacked = idx in attacked_set
        n_mal       = int(len(clean) * malicious_pct / 100) if is_attacked else 0
        combined    = clean + (generate_malicious_flows(n_mal, clean) if is_attacked else [])

        sig = epoch_signal(combined, bl_dists, bl_src_set)

        alarms = {}
        any_alarm = False
        for key, state in cusum_states.items():
            fired = state.update(sig[key])
            alarms[key] = {"alarm": fired, **state.snapshot(),
                           "signal_value": sig[key]}
            if fired:
                any_alarm = True

        epoch_results.append({
            "epoch_index": idx,
            "epoch_name": os.path.splitext(os.path.basename(epoch_files[idx]))[0],
            "attacked": is_attacked,
            "clean_flows_count": len(clean),
            "malicious_flows_count": n_mal,
            "total_flows_count": len(combined),
            "signal": sig,
            "cusum": alarms,
            "cusum_alarm": any_alarm,
        })
    return epoch_results


def run_single_configuration(directory, baseline_pct, malicious_pct,
                              dataset_name, out_dir, k_factor=0.5, h_factor=4.0):

    epoch_files = _epoch_files(directory)
    n_epochs    = len(epoch_files)
    n_baseline  = max(1, int(n_epochs * baseline_pct / 100))
    bl_idx      = list(range(n_baseline))
    atk_idx     = list(range(n_baseline, n_epochs))

    print(f"\n  [CUSUM] Baseline={baseline_pct}%, Malicious={malicious_pct}%")

    bl_flows_list = [parse_flows(epoch_files[i]) for i in bl_idx
                     if parse_flows(epoch_files[i])]
    bl_dists  = build_baseline_dists(bl_flows_list)
    bl_src_set = bl_dists.get("all_src_ips", set())

    # Calibrate CUSUM from baseline signals
    bl_signals = [epoch_signal(flows, bl_dists, bl_src_set)
                  for flows in bl_flows_list]
    cusum_states = build_cusum_states(bl_signals, k_factor, h_factor)

    epoch_results = _run_epochs(epoch_files, atk_idx, set(atk_idx),
                                 malicious_pct, bl_dists, bl_src_set, cusum_states)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "CUSUM Change-Point (JS + NewIP + TopK)",
            "total_epochs": n_epochs,
            "baseline_epoch_count": n_baseline,
            "baseline_percentage": baseline_pct,
            "malicious_percentage": malicious_pct,
            "k_factor": k_factor,
            "h_factor": h_factor,
            "sparse_mode": False,
        },
        "baseline": {
            key: {"mean": round(s.mean, 6), "std": round(s.std, 6),
                  "k": round(s.k, 6), "h": round(s.h, 6)}
            for key, s in cusum_states.items()
        },
        "epochs": epoch_results,
    }

    out_path = os.path.join(out_dir,
        f"{dataset_name}_{baseline_pct}_{malicious_pct}_cusum.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {os.path.basename(out_path)}")
    return results


def run_sparse_configuration(directory, baseline_pct, malicious_pct,
                              dataset_name, out_dir,
                              attack_prob=0.3, window_size=10,
                              k_factor=0.5, h_factor=4.0):

    epoch_files = _epoch_files(directory)
    n_epochs   = len(epoch_files)
    n_baseline = max(1, int(n_epochs * baseline_pct / 100))
    bl_idx     = list(range(n_baseline))
    atk_pool   = list(range(n_baseline, n_epochs))

    windows    = [atk_pool[s: s+window_size] for s in range(0, len(atk_pool), window_size)]
    attacked_w = [i for i in range(len(windows)) if random.random() < attack_prob]
    attacked_set = {ep for wi in attacked_w for ep in windows[wi]}

    print(f"\n  [CUSUM SPARSE] Windows: {len(windows)}, Attacked: {len(attacked_w)}")

    bl_flows_list = [parse_flows(epoch_files[i]) for i in bl_idx
                     if parse_flows(epoch_files[i])]
    bl_dists   = build_baseline_dists(bl_flows_list)
    bl_src_set = bl_dists.get("all_src_ips", set())
    bl_signals = [epoch_signal(flows, bl_dists, bl_src_set) for flows in bl_flows_list]
    cusum_states = build_cusum_states(bl_signals, k_factor, h_factor)

    epoch_results = _run_epochs(epoch_files, atk_pool, attacked_set,
                                 malicious_pct, bl_dists, bl_src_set, cusum_states)

    attacked_window_info = [
        {"window_index": wi, "epoch_indices": windows[wi],
         "epoch_range": [windows[wi][0], windows[wi][-1]] if windows[wi] else []}
        for wi in attacked_w
    ]

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "CUSUM Change-Point (JS + NewIP + TopK)",
            "sparse_mode": True,
            "sparse_attack_prob": attack_prob,
            "sparse_window_size": window_size,
            "k_factor": k_factor, "h_factor": h_factor,
            "attacked_windows": attacked_window_info,
        },
        "baseline": {
            key: {"mean": round(s.mean, 6), "std": round(s.std, 6),
                  "k": round(s.k, 6), "h": round(s.h, 6)}
            for key, s in cusum_states.items()
        },
        "epochs": epoch_results,
    }

    out_path = os.path.join(out_dir,
        f"{dataset_name}_{baseline_pct}_{malicious_pct}_cusum_sparse.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {os.path.basename(out_path)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CUSUM Change-Point Detector")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--attack-prob", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--h-factor", type=float, default=4.0,
                        help="CUSUM alarm threshold = h_factor * σ (default: 4.0)")
    parser.add_argument("--k-factor", type=float, default=0.5,
                        help="CUSUM slack = k_factor * σ (default: 0.5)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"CUSUM CHANGE-POINT DETECTOR  [{'SPARSE' if args.sparse else 'FULL'} MODE]")
    print("=" * 80)

    directory    = args.directory or input("\nEpoch directory: ").strip()
    dataset_name = os.path.basename(os.path.normpath(directory))
    out_dir      = os.path.join(directory, "json")
    os.makedirs(out_dir, exist_ok=True)

    BASELINE_PERCENTAGES  = [5.0, 10.0]
    MALICIOUS_PERCENTAGES = [0.0, 1.0, 2.0, 5.0]

    total = len(BASELINE_PERCENTAGES) * len(MALICIOUS_PERCENTAGES)
    n = 0
    for bp in BASELINE_PERCENTAGES:
        for mp in MALICIOUS_PERCENTAGES:
            n += 1
            print(f"\n[{n}/{total}]", "-" * 60)
            try:
                if args.sparse:
                    run_sparse_configuration(directory, bp, mp, dataset_name,
                                             out_dir, args.attack_prob,
                                             args.window_size,
                                             args.k_factor, args.h_factor)
                else:
                    run_single_configuration(directory, bp, mp, dataset_name,
                                             out_dir, args.k_factor, args.h_factor)
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"DONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
