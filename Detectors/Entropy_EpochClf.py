from typing import Tuple, Dict, List
from dataclasses import dataclass, field
import zlib
import hashlib
import math
import os
import random
import glob
import json

# Flow
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
                # Expected unique elements in this bucket
                # Using: w × (1 - e^(-count/w))
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
                # Shannon entropy: -p * log2(p)
                # Note: log2(p) is negative when 0 < p < 1, so we add it
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
        # 2^entropy gives us the "effective number" of elements
        # Dividing by cardinality gives us a uniformity metric
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

def calculate_all_entropies(flows: List[Flow], width: int = 1024, depth: int = 5) -> EntropyMetrics:
    entropy_calc = Entropy(width=width, depth=depth)
    
    for flow in flows:
        entropy_calc.add_flow(flow)
    
    unique_src_ips = list(set(flow.src_ip for flow in flows))
    unique_dst_ips = list(set(flow.dst_ip for flow in flows))
    unique_src_ports = list(set(flow.src_port for flow in flows))
    unique_dst_ports = list(set(flow.dst_port for flow in flows))
    
    src_ip_entropy, src_ip_cardinality, _ = entropy_calc.get_src_ip_entropy(unique_src_ips)
    dst_ip_entropy, dst_ip_cardinality, _ = entropy_calc.get_dst_ip_entropy(unique_dst_ips)
    src_port_entropy, src_port_cardinality, _ = entropy_calc.get_src_port_entropy(unique_src_ports)
    dst_port_entropy, dst_port_cardinality, _ = entropy_calc.get_dst_port_entropy(unique_dst_ports)
    
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
    )

def generate_malicious_flows(num_flows: int, base_flows: List[Flow]) -> List[Flow]:
    print(f"\nGenerating {num_flows} malicious flows...")
    
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

def print_entropy_metrics(metrics: EntropyMetrics, label: str):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"\nFlow Statistics:")
    print(f"  Total Flows: {metrics.total_flows:,}")
    print(f"  Unique Source IPs: {metrics.unique_src_ips:,}")
    print(f"  Unique Destination IPs: {metrics.unique_dst_ips:,}")
    print(f"  Unique Source Ports: {metrics.unique_src_ports:,}")
    print(f"  Unique Destination Ports: {metrics.unique_dst_ports:,}")
    
    print(f"\nSource IP Metrics:")
    print(f"  Entropy: {metrics.src_ip_entropy:.4f} bits")
    print(f"  Estimated Cardinality: {metrics.src_ip_cardinality:.2f}")
    print(f"  Uniformity: {metrics.src_ip_uniformity:.4f}")
    
    print(f"\nDestination IP Metrics:")
    print(f"  Entropy: {metrics.dst_ip_entropy:.4f} bits")
    print(f"  Estimated Cardinality: {metrics.dst_ip_cardinality:.2f}")
    print(f"  Uniformity: {metrics.dst_ip_uniformity:.4f}")
    
    print(f"\nSource Port Metrics:")
    print(f"  Entropy: {metrics.src_port_entropy:.4f} bits")
    print(f"  Estimated Cardinality: {metrics.src_port_cardinality:.2f}")
    
    print(f"\nDestination Port Metrics:")
    print(f"  Entropy: {metrics.dst_port_entropy:.4f} bits")
    print(f"  Estimated Cardinality: {metrics.dst_port_cardinality:.2f}")
    
    print(f"\nSource/Destination Ratio Metrics:")
    print(f"  Source/Destination Ratio: {metrics.src_dst_ratio:.4f}")
    
def print_entropy_differences(baseline: EntropyMetrics, with_malicious: EntropyMetrics):
    print(f"\n{'='*80}")
    print("ENTROPY DIFFERENCES (After Malicious Flows)")
    print(f"{'='*80}")
    
    print(f"\nFlow Count Changes:")
    print(f"  Baseline Flows: {baseline.total_flows:,}")
    print(f"  With Malicious Flows: {with_malicious.total_flows:,}")
    print(f"  Difference: +{with_malicious.total_flows - baseline.total_flows:,}")
    
    print(f"\nSource IP Entropy:")
    print(f"  Baseline: {baseline.src_ip_entropy:.4f} bits")
    print(f"  With Malicious: {with_malicious.src_ip_entropy:.4f} bits")
    print(f"  Difference: {with_malicious.src_ip_entropy - baseline.src_ip_entropy:+.4f} bits")
    print(f"  Percent Change: {((with_malicious.src_ip_entropy - baseline.src_ip_entropy) / baseline.src_ip_entropy * 100) if baseline.src_ip_entropy > 0 else 0:+.2f}%")
    
    print(f"\nDestination IP Entropy:")
    print(f"  Baseline: {baseline.dst_ip_entropy:.4f} bits")
    print(f"  With Malicious: {with_malicious.dst_ip_entropy:.4f} bits")
    print(f"  Difference: {with_malicious.dst_ip_entropy - baseline.dst_ip_entropy:+.4f} bits")
    print(f"  Percent Change: {((with_malicious.dst_ip_entropy - baseline.dst_ip_entropy) / baseline.dst_ip_entropy * 100) if baseline.dst_ip_entropy > 0 else 0:+.2f}%")
    
    print(f"\nSource Port Entropy:")
    print(f"  Baseline: {baseline.src_port_entropy:.4f} bits")
    print(f"  With Malicious: {with_malicious.src_port_entropy:.4f} bits")
    print(f"  Difference: {with_malicious.src_port_entropy - baseline.src_port_entropy:+.4f} bits")
    print(f"  Percent Change: {((with_malicious.src_port_entropy - baseline.src_port_entropy) / baseline.src_port_entropy * 100) if baseline.src_port_entropy > 0 else 0:+.2f}%")
    
    print(f"\nDestination Port Entropy:")
    print(f"  Baseline: {baseline.dst_port_entropy:.4f} bits")
    print(f"  With Malicious: {with_malicious.dst_port_entropy:.4f} bits")
    print(f"  Difference: {with_malicious.dst_port_entropy - baseline.dst_port_entropy:+.4f} bits")
    print(f"  Percent Change: {((with_malicious.dst_port_entropy - baseline.dst_port_entropy) / baseline.dst_port_entropy * 100) if baseline.dst_port_entropy > 0 else 0:+.2f}%")
    
    print(f"\nCardinality Changes:")
    print(f"  Source IP Cardinality Change: {with_malicious.src_ip_cardinality - baseline.src_ip_cardinality:+.2f}")
    print(f"  Destination IP Cardinality Change: {with_malicious.dst_ip_cardinality - baseline.dst_ip_cardinality:+.2f}")
    print(f"  Source Port Cardinality Change: {with_malicious.src_port_cardinality - baseline.src_port_cardinality:+.2f}")
    print(f"  Destination Port Cardinality Change: {with_malicious.dst_port_cardinality - baseline.dst_port_cardinality:+.2f}")
    
    print(f"\nSrc/Dst Ratio Changes:")
    print(f"  Baseline Ratio: {baseline.src_dst_ratio:.4f}")
    print(f"  With Malicious Ratio: {with_malicious.src_dst_ratio:.4f}")
    print(f"  Difference: {with_malicious.src_dst_ratio - baseline.src_dst_ratio:+.4f}")
    print(f"  Percent Change: {((with_malicious.src_dst_ratio - baseline.src_dst_ratio) / baseline.src_dst_ratio * 100) if baseline.src_dst_ratio > 0 else 0:+.2f}%")


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
    }

def compute_difference_dict(baseline: dict, polluted: dict) -> dict:
    diff = {}
    for key in baseline:
        if isinstance(baseline[key], (int, float)):
            diff[key] = round(polluted[key] - baseline[key], 6)
        else:
            diff[key] = None
    return diff

def main():
    import sys
    
    print("="*80)
    print("EPOCH-WISE ENTROPY ANALYSIS WITH MALICIOUS FLOW INJECTION")
    print("="*80)
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("\nEnter the directory path containing epoch flow files: ").strip()
    
    if not os.path.isdir(directory):
        print(f"\nError: Directory not found: {directory}")
        return
    
    # Get malicious percentage
    if len(sys.argv) > 2:
        try:
            malicious_percentage = float(sys.argv[2])
        except ValueError:
            print("Invalid percentage provided. Using default 10%")
            malicious_percentage = 10.0
    else:
        malicious_input = input("Enter percentage of malicious flows to inject (e.g., 5 for 5%): ").strip()
        try:
            malicious_percentage = float(malicious_input)
        except ValueError:
            print("Invalid input. Using default 10%")
            malicious_percentage = 10.0
    
    # Discover epoch files
    epoch_files = sorted(glob.glob(os.path.join(directory, 'epoch_*.txt')))
    if not epoch_files:
        # Fallback: try all files
        epoch_files = sorted(
            f for f in glob.glob(os.path.join(directory, '*'))
            if os.path.isfile(f)
        )
    
    if not epoch_files:
        print("\nError: No epoch files found. Exiting.")
        return
    
    print(f"\nFound {len(epoch_files)} epoch files in: {directory}")
    print(f"Malicious traffic injection: {malicious_percentage}%")
    
    all_results = []
    
    random.seed(42)
    for idx, epoch_file in enumerate(epoch_files):
        epoch_name = os.path.splitext(os.path.basename(epoch_file))[0]
        
        # Parse flows for this epoch
        flows = parse_flows(epoch_file)
        if not flows:
            print(f"  [{idx+1}/{len(epoch_files)}] {epoch_name}: No flows, skipping.")
            continue
        
        # Baseline entropies for this epoch
        baseline_metrics = calculate_all_entropies(flows)
        baseline_dict = metrics_to_dict(baseline_metrics)
        
        # Generate malicious flows proportional to THIS epoch's flow count
        num_malicious = int(len(flows) * (malicious_percentage / 100))
        malicious_flows = generate_malicious_flows(num_malicious, flows)
        
        # Polluted entropies for this epoch
        combined_flows = flows + malicious_flows
        polluted_metrics = calculate_all_entropies(combined_flows)
        polluted_dict = metrics_to_dict(polluted_metrics)
        
        # Differences
        diff_dict = compute_difference_dict(baseline_dict, polluted_dict)
        
        epoch_result = {
            "epoch": epoch_name,
            "epoch_index": idx,
            "total_flows": len(flows),
            "malicious_flows_added": num_malicious,
            "malicious_percent": malicious_percentage,
            "baseline": baseline_dict,
            "polluted": polluted_dict,
            "difference": diff_dict,
        }
        
        all_results.append(epoch_result)
        
        print(f"  [{idx+1}/{len(epoch_files)}] {epoch_name}: "
              f"{len(flows)} flows, +{num_malicious} malicious | "
              f"src_ip diff = {diff_dict['src_ip_entropy']:+.4f}, "
              f"dst_ip diff = {diff_dict['dst_ip_entropy']:+.4f}")
    
    # Save to JSON
    folder_name = os.path.basename(os.path.normpath(directory))
    output_filename = f"entropy_epoch_analysis_{folder_name}_{malicious_percentage}pct.json"
    output_path = os.path.join(directory, '..', output_filename)
    output_path = os.path.normpath(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Epochs processed: {len(all_results)}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()