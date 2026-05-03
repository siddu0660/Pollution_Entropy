"""
Entropy_HHH.py  –  Heavy Hitter + Sketch Chi-Squared Attack Detector
======================================================================

Key Insight
-----------
Both Entropy and JS-divergence are *global* distribution metrics.  When an
attack injects flows from a small number of dominating (heavy-hitter) source
IPs, the global distribution barely moves because those sources may look
statistically like normal top-talkers.

This script uses two complementary ideas that are much more sensitive:

1. **Heavy Hitter fraction** (via CMS top-count approximation)
   Track how much of total traffic is captured by the top-K src/dst IPs.
   During an attack the top-K fraction suddenly increases (traffic skew).

2. **CMS sketch Chi-Squared distance** vs. baseline
   Instead of JS divergence, compute the χ² statistic between the per-feature
   CMS sketch rows of the epoch and baseline.  χ² is provably more sensitive
   to sparse heavy hitters than KL or JS, and avoids the log(0) edge cases.

   χ²(P, Q) = Σ_i (P_i – Q_i)² / (Q_i + ε)

Combined into a single `combined_score` (mean of normalised χ² and HH skew
z-score) → provides a strong, robust signal even at 1–2 % attack traffic.

Usage
-----
  python Entropy_HHH.py <directory> [--sparse] [--attack-prob 0.3] \
                         [--window-size 10] [--topk 5]
"""

from typing import List, Dict, Tuple
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
# Count-Min Sketch with sketch-vector export
# ---------------------------------------------------------------------------

@dataclass
class CMS:
    width: int
    depth: int
    table: List[List[int]] = field(default_factory=list, init=False)
    total: int = field(default=0, init=False)

    def __post_init__(self):
        self.table = [[0] * self.width for _ in range(self.depth)]

    def _hash(self, item: str, row: int) -> int:
        return zlib.crc32(f"{item}_{row}".encode()) % self.width

    def add(self, item: str, cnt: int = 1) -> None:
        for r in range(self.depth):
            self.table[r][self._hash(item, r)] += cnt
        self.total += cnt

    def estimate(self, item: str) -> int:
        return min(self.table[r][self._hash(item, r)] for r in range(self.depth))

    def sketch_row(self, row: int = 0) -> List[float]:
        """Return probability vector from sketch row `row`."""
        if self.total == 0:
            return [0.0] * self.width
        return [c / self.total for c in self.table[row]]


# ---------------------------------------------------------------------------
# Chi-Squared distance between two probability vectors
# ---------------------------------------------------------------------------

def chi_squared(p: List[float], q: List[float], epsilon: float = 1e-10) -> float:
    """Σ (p_i - q_i)² / (q_i + ε)  — one-sided, sensitive to heavy hitters."""
    return sum((pi - qi) ** 2 / (qi + epsilon) for pi, qi in zip(p, q))


# ---------------------------------------------------------------------------
# Top-K heavy hitter fraction
# ---------------------------------------------------------------------------

def topk_fraction(flows: List[Flow], feature: str, k: int = 5) -> float:
    """Fraction of flows accounted for by the top-K values of `feature`."""
    counts: Dict[str, int] = defaultdict(int)
    for flow in flows:
        counts[getattr(flow, feature)] += 1
    if not flows:
        return 0.0
    top_k = sorted(counts.values(), reverse=True)[:k]
    return sum(top_k) / len(flows)


# ---------------------------------------------------------------------------
# Baseline sketch (aggregated over all baseline epochs)
# ---------------------------------------------------------------------------

def build_baseline_cms(baseline_flows_list: List[List[Flow]],
                        width: int = 512,
                        depth: int = 5
                        ) -> Dict[str, CMS]:
    """One CMS per feature, trained on all baseline flows."""
    cms_map = {
        "src_ip":   CMS(width, depth),
        "dst_ip":   CMS(width, depth),
        "src_port": CMS(width, depth),
        "dst_port": CMS(width, depth),
    }
    for flows in baseline_flows_list:
        for flow in flows:
            cms_map["src_ip"].add(flow.src_ip)
            cms_map["dst_ip"].add(flow.dst_ip)
            cms_map["src_port"].add(flow.src_port)
            cms_map["dst_port"].add(flow.dst_port)
    return cms_map


# ---------------------------------------------------------------------------
# Per-epoch metrics
# ---------------------------------------------------------------------------

@dataclass
class HHMetrics:
    total_flows: int
    # χ² distance of each feature vs baseline sketch
    src_ip_chi2: float
    dst_ip_chi2: float
    src_port_chi2: float
    dst_port_chi2: float
    combined_chi2: float
    # Top-K fraction per feature
    src_ip_topk: float
    dst_ip_topk: float
    # Z-scores vs baseline topk distribution
    src_topk_zscore: float = 0.0
    dst_topk_zscore: float = 0.0
    chi2_zscore: float = 0.0
    combined_score: float = 0.0


def compute_hhmetrics(flows: List[Flow],
                      baseline_cms: Dict[str, CMS],
                      topk: int = 5,
                      width: int = 512,
                      depth: int = 5) -> HHMetrics:
    epoch_cms = {
        "src_ip":   CMS(width, depth),
        "dst_ip":   CMS(width, depth),
        "src_port": CMS(width, depth),
        "dst_port": CMS(width, depth),
    }
    for flow in flows:
        epoch_cms["src_ip"].add(flow.src_ip)
        epoch_cms["dst_ip"].add(flow.dst_ip)
        epoch_cms["src_port"].add(flow.src_port)
        epoch_cms["dst_port"].add(flow.dst_port)

    chi2_vals = {}
    for feat in ["src_ip", "dst_ip", "src_port", "dst_port"]:
        ep_row = epoch_cms[feat].sketch_row(0)
        bl_row = baseline_cms[feat].sketch_row(0)
        chi2_vals[feat] = round(chi_squared(ep_row, bl_row), 6)

    combined_chi2 = round(statistics.mean(chi2_vals.values()), 6)

    src_topk = round(topk_fraction(flows, "src_ip", topk), 6)
    dst_topk = round(topk_fraction(flows, "dst_ip", topk), 6)

    return HHMetrics(
        total_flows=len(flows),
        src_ip_chi2=chi2_vals["src_ip"],
        dst_ip_chi2=chi2_vals["dst_ip"],
        src_port_chi2=chi2_vals["src_port"],
        dst_port_chi2=chi2_vals["dst_port"],
        combined_chi2=combined_chi2,
        src_ip_topk=src_topk,
        dst_ip_topk=dst_topk,
    )


def calibrate_baseline_stats(baseline_flows_list: List[List[Flow]],
                               baseline_cms: Dict[str, CMS],
                               topk: int = 5,
                               width: int = 512,
                               depth: int = 5) -> dict:
    records = [compute_hhmetrics(flows, baseline_cms, topk, width, depth)
               for flows in baseline_flows_list if flows]

    def ms(vals):
        m = statistics.mean(vals) if vals else 0.0
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    mc, sc = ms([r.combined_chi2 for r in records])
    ms_tk, ss_tk = ms([r.src_ip_topk for r in records])
    md_tk, sd_tk = ms([r.dst_ip_topk for r in records])

    return {
        "mean_combined_chi2": mc, "std_combined_chi2": sc,
        "mean_src_topk": ms_tk,   "std_src_topk": ss_tk,
        "mean_dst_topk": md_tk,   "std_dst_topk": sd_tk,
    }


def attach_zscores(m: HHMetrics, bl: dict) -> HHMetrics:
    def zs(v, mean, std):
        return (v - mean) / std if std > 1e-12 else 0.0

    m.chi2_zscore      = round(zs(m.combined_chi2, bl["mean_combined_chi2"], bl["std_combined_chi2"]), 4)
    m.src_topk_zscore  = round(zs(m.src_ip_topk,   bl["mean_src_topk"],      bl["std_src_topk"]),      4)
    m.dst_topk_zscore  = round(zs(m.dst_ip_topk,   bl["mean_dst_topk"],      bl["std_dst_topk"]),      4)
    m.combined_score   = round(statistics.mean([
        abs(m.chi2_zscore), abs(m.src_topk_zscore), abs(m.dst_topk_zscore)
    ]), 4)
    return m


def metrics_to_dict(m: HHMetrics) -> dict:
    return {
        "total_flows": m.total_flows,
        "src_ip_chi2": m.src_ip_chi2,
        "dst_ip_chi2": m.dst_ip_chi2,
        "src_port_chi2": m.src_port_chi2,
        "dst_port_chi2": m.dst_port_chi2,
        "combined_chi2": m.combined_chi2,
        "src_ip_topk": m.src_ip_topk,
        "dst_ip_topk": m.dst_ip_topk,
        "chi2_zscore": m.chi2_zscore,
        "src_topk_zscore": m.src_topk_zscore,
        "dst_topk_zscore": m.dst_topk_zscore,
        "combined_score": m.combined_score,
    }


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


def run_single_configuration(directory: str, baseline_pct: float,
                              malicious_pct: float,
                              dataset_name: str, out_dir: str,
                              topk: int = 5, width: int = 512, depth: int = 5):

    epoch_files = _epoch_files(directory)
    n_epochs    = len(epoch_files)
    n_baseline  = max(1, int(n_epochs * baseline_pct / 100))
    bl_idx      = list(range(n_baseline))
    atk_idx     = list(range(n_baseline, n_epochs))

    print(f"\n  [HHH] Baseline={baseline_pct}%, Malicious={malicious_pct}%")

    bl_flows_list = [parse_flows(epoch_files[i]) for i in bl_idx
                     if parse_flows(epoch_files[i])]
    bl_cms  = build_baseline_cms(bl_flows_list, width, depth)
    bl_stats = calibrate_baseline_stats(bl_flows_list, bl_cms, topk, width, depth)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "Heavy Hitter + Chi-Squared Sketch",
            "topk": topk, "width": width, "depth": depth,
            "total_epochs": n_epochs,
            "baseline_epoch_count": n_baseline,
            "baseline_percentage": baseline_pct,
            "malicious_percentage": malicious_pct,
            "sparse_mode": False,
        },
        "baseline": {k: round(v, 6) for k, v in bl_stats.items()},
        "epochs": [],
    }

    for idx in atk_idx:
        clean = parse_flows(epoch_files[idx])
        if not clean:
            continue
        n_mal    = int(len(clean) * malicious_pct / 100)
        combined = clean + generate_malicious_flows(n_mal, clean)
        m = attach_zscores(compute_hhmetrics(combined, bl_cms, topk, width, depth), bl_stats)
        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": os.path.splitext(os.path.basename(epoch_files[idx]))[0],
            "attacked": True,
            "clean_flows_count": len(clean),
            "malicious_flows_count": n_mal,
            "total_flows_count": len(combined),
            "metrics": metrics_to_dict(m),
        })

    out_path = os.path.join(out_dir,
        f"{dataset_name}_{baseline_pct}_{malicious_pct}_hhh.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {os.path.basename(out_path)}")
    return results


def run_sparse_configuration(directory: str, baseline_pct: float,
                              malicious_pct: float,
                              dataset_name: str, out_dir: str,
                              attack_prob: float = 0.3,
                              window_size: int = 10,
                              topk: int = 5, width: int = 512, depth: int = 5):

    epoch_files = _epoch_files(directory)
    n_epochs   = len(epoch_files)
    n_baseline = max(1, int(n_epochs * baseline_pct / 100))
    bl_idx     = list(range(n_baseline))
    atk_pool   = list(range(n_baseline, n_epochs))

    windows     = [atk_pool[s: s + window_size]
                   for s in range(0, len(atk_pool), window_size)]
    attacked_w  = [i for i in range(len(windows)) if random.random() < attack_prob]
    attacked_set = {ep for wi in attacked_w for ep in windows[wi]}

    print(f"\n  [HHH SPARSE] Windows: {len(windows)}, Attacked: {len(attacked_w)}")

    bl_flows_list = [parse_flows(epoch_files[i]) for i in bl_idx
                     if parse_flows(epoch_files[i])]
    bl_cms   = build_baseline_cms(bl_flows_list, width, depth)
    bl_stats  = calibrate_baseline_stats(bl_flows_list, bl_cms, topk, width, depth)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "method": "Heavy Hitter + Chi-Squared Sketch",
            "sparse_mode": True,
            "sparse_attack_prob": attack_prob,
            "sparse_window_size": window_size,
        },
        "baseline": {k: round(v, 6) for k, v in bl_stats.items()},
        "epochs": [],
    }

    for idx in atk_pool:
        clean = parse_flows(epoch_files[idx])
        if not clean:
            continue
        is_attacked = idx in attacked_set
        n_mal       = int(len(clean) * malicious_pct / 100) if is_attacked else 0
        combined    = clean + (generate_malicious_flows(n_mal, clean) if is_attacked else [])
        m = attach_zscores(compute_hhmetrics(combined, bl_cms, topk, width, depth), bl_stats)
        results["epochs"].append({
            "epoch_index": idx,
            "epoch_name": os.path.splitext(os.path.basename(epoch_files[idx]))[0],
            "attacked": is_attacked,
            "clean_flows_count": len(clean),
            "malicious_flows_count": n_mal,
            "total_flows_count": len(combined),
            "metrics": metrics_to_dict(m),
        })

    out_path = os.path.join(out_dir,
        f"{dataset_name}_{baseline_pct}_{malicious_pct}_hhh_sparse.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {os.path.basename(out_path)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Heavy Hitter + Chi-Squared Attack Detector")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--attack-prob", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of top heavy hitters to consider (default: 5)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"HEAVY HITTER + CHI-SQUARED DETECTOR  [{'SPARSE' if args.sparse else 'FULL'} MODE]")
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
                                             args.window_size, args.topk)
                else:
                    run_single_configuration(directory, bp, mp,
                                             dataset_name, out_dir, args.topk)
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"DONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
