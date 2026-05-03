from typing import Tuple, Dict
from dataclasses import dataclass, field
from typing import List
import zlib
import hashlib
import random
import math

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
    
    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge sketches with different dimensions")
        
        merged = CountMinSketch(width=self.width, depth=self.depth)
        for i in range(self.depth):
            for j in range(self.width):
                merged.sketch[i][j] = self.sketch[i][j] + other.sketch[i][j]
        merged.total_count = self.total_count + other.total_count
        return merged
    
# Entropy
@dataclass
class Entropy:
    width: int
    depth: int
    src_ip_cms: CountMinSketch = field(init=False)
    dst_ip_cms: CountMinSketch = field(init=False)
    src_port_cms: CountMinSketch = field(init=False)
    dst_port_cms: CountMinSketch = field(init=False)
    log_table: List[float] = field(init=False, repr=False)
    
    def __post_init__(self):
        self.src_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_ip_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.src_port_cms = CountMinSketch(width=self.width, depth=self.depth)
        self.dst_port_cms = CountMinSketch(width=self.width, depth=self.depth)
        
        self.log_table = [0.0] * 256
        for i in range(1, 256):
            self.log_table[i] = math.log2(i / 256.0)
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
                index = int(prob * 256)
                if index >= 256:
                    index = 255
                if index > 0:
                    entropy -= prob * self.log_table[index]
            
        cardinality = cms.estimate_cardinality_mle()
        return entropy, cardinality, non_zero_items
    
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
    
    def get_src_ip_entropy(self, unique_src_ips: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_ip_cms, unique_src_ips)
    
    def get_dst_ip_entropy(self, unique_dst_ips: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.dst_ip_cms, unique_dst_ips)
    
    def get_src_port_entropy(self, unique_src_ports: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.src_port_cms, unique_src_ports)
    
    def get_dst_port_entropy(self, unique_dst_ports: List[str]) -> Tuple[float, float, int]:
        return self.calculate_entropy(self.dst_port_cms, unique_dst_ports)

def generate_synthetic_flows(num_flows: int) -> List[Flow]:
    flows = []
    random.seed(42)
    
    for i in range(num_flows):
        src_ip = f"{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}"
        
        dst_ip = f"{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}"
        
        src_port = f"{random.randint(0, 65535):05d}"
        
        dst_port_val = random.choice([80, 443, 808, 22, 53, 330, 543, random.randint(0, 65535)])
        dst_port = f"{dst_port_val % 65535:05d}"
        
        protocol = f"{random.choice([6, 17]):02d}"
        
        flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    return flows

def generate_attack_flows(num_flows: int, attack_type: str = "ddos") -> List[Flow]:
    flows = []
    
    if attack_type == "ddos":
        # DDoS: Many sources, few destinations, high entropy
        num_sources = num_flows // 10  # Many unique sources
        target_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        
        for i in range(num_flows):
            src_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = "00080"  # Port 80
            protocol = "06"  # TCP
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    elif attack_type == "portscan":
        # Port scan: Few sources, many destination ports
        attacker_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        target_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        
        for i in range(num_flows):
            src_ip = attacker_ip
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = f"{random.randint(1, 65535):05d}"  # Scanning all ports
            protocol = "06"  # TCP
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    elif attack_type == "legitimate_spike":
        # Legitimate spike: Moderate sources, skewed distribution
        num_sources = max(50, num_flows // 100)  # Fewer sources
        
        # Power law distribution: few heavy users, many light users
        sources = []
        for i in range(num_sources):
            src_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            weight = int((num_sources - i) ** 1.5)
            sources.extend([src_ip] * weight)
        
        for i in range(num_flows):
            src_ip = random.choice(sources)
            dst_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = random.choice(["00080", "00443"])  # HTTP/HTTPS
            protocol = "06"  # TCP
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))

    else:  # normal traffic
        for i in range(num_flows):
            src_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            dst_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port_val = random.choice([80, 443, 22, 53, 8080])
            dst_port = f"{dst_port_val:05d}"
            protocol = random.choice(["06", "17"])
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    return flows

@dataclass
class TrafficCluster:
    cluster_id: str
    flows: List[Flow] = field(default_factory=list)
    entropy_tracker: Entropy = field(init=False)
    
    def __post_init__(self):
        self.entropy_tracker = Entropy(width=1024, depth=4)
    
    def add_flow(self, flow: Flow):
        self.flows.append(flow)
        self.entropy_tracker.add_flow(flow)
    
    def get_metrics(self) -> Dict:
        unique_src_ips = list(set(flow.src_ip for flow in self.flows))
        unique_dst_ips = list(set(flow.dst_ip for flow in self.flows))
        unique_src_ports = list(set(flow.src_port for flow in self.flows))
        unique_dst_ports = list(set(flow.dst_port for flow in self.flows))
        
        unique_flows = set()
        for flow in self.flows:
            flow_tuple = (flow.src_ip, flow.dst_ip, flow.src_port, flow.dst_port, flow.protocol)
            unique_flows.add(flow_tuple)
        
        num_unique_flows = len(unique_flows)
        repetition_ratio = len(self.flows) / num_unique_flows if num_unique_flows > 0 else 1.0
        
        src_ip_entropy, src_ip_card, _ = self.entropy_tracker.get_src_ip_entropy(unique_src_ips)
        dst_ip_entropy, dst_ip_card, _ = self.entropy_tracker.get_dst_ip_entropy(unique_dst_ips)
        src_port_entropy, src_port_card, _ = self.entropy_tracker.get_src_port_entropy(unique_src_ports)
        dst_port_entropy, dst_port_card, _ = self.entropy_tracker.get_dst_port_entropy(unique_dst_ports)
        
        src_uniformity = Entropy.calculate_effectiveness(src_ip_entropy, src_ip_card)
        dst_uniformity = Entropy.calculate_effectiveness(dst_ip_entropy, dst_ip_card)
        
        return {
            'cluster_id': self.cluster_id,
            'total_flows': len(self.flows),
            'unique_flows': num_unique_flows,
            'repetition_ratio': repetition_ratio,
            'actual_unique_src_ips': len(unique_src_ips),
            'actual_unique_dst_ips': len(unique_dst_ips),
            'src_ip_entropy': src_ip_entropy,
            'src_ip_cardinality': src_ip_card,
            'src_ip_uniformity': src_uniformity,
            'dst_ip_entropy': dst_ip_entropy,
            'dst_ip_cardinality': dst_ip_card,
            'dst_ip_uniformity': dst_uniformity,
            'src_port_entropy': src_port_entropy,
            'src_port_cardinality': src_port_card,
            'dst_port_entropy': dst_port_entropy,
            'dst_port_cardinality': dst_port_card,
            'src_dst_ratio': src_ip_card / dst_ip_card if dst_ip_card > 0 else 0,
        }
    
    def classify_traffic(self, metrics: Dict) -> Tuple[str, float]:
        """
        1. Port Scan: Low source diversity + High dest port diversity
        2. DDoS (Single Target): Many sources + Single/Few destinations
        3. DDoS (High Uniformity): High source diversity + Low dest diversity + High uniformity
        4. Legitimate Spike: Moderate diversity + Low uniformity (skewed)
        5. Normal Traffic: Balanced, moderate metrics
        """
        src_entropy = metrics['src_ip_entropy']
        dst_entropy = metrics['dst_ip_entropy']
        dst_port_entropy = metrics['dst_port_entropy']
        src_card = metrics['src_ip_cardinality']
        dst_card = metrics['dst_ip_cardinality']
        src_uniformity = metrics['src_ip_uniformity']
        dst_uniformity = metrics['dst_ip_uniformity']
        src_dst_ratio = metrics['src_dst_ratio']
        src_port_entropy = metrics['src_port_entropy']
        dst_port_card = metrics['dst_port_cardinality']
        total_flows = metrics['total_flows']
        actual_unique_src = metrics['actual_unique_src_ips']
        actual_unique_dst = metrics['actual_unique_dst_ips']
        unique_flows = metrics.get('unique_flows', total_flows)
        repetition_ratio = metrics.get('repetition_ratio', 1.0)
        base_confidence = 0.60
        
        # FLOODING/REPETITIVE ATTACK: Very high flow repetition (same flows repeated many times)
        # This is a strong indicator of DDoS, flooding, or replay attacks
        # Key: repetition_ratio > 10 means flows are repeated on average 10+ times
        if repetition_ratio > 10 and total_flows > 1000:
            confidence = base_confidence + 0.35
            if repetition_ratio > 20:  # Extremely repetitive
                confidence += 0.05
            if actual_unique_src > 50:  # Many sources doing repetition
                confidence += 0.03
                return "DDoS Attack (Flooding)", min(confidence, 0.98)
            else:
                return "Flooding Attack", min(confidence, 0.95)
        
        # PORT SCAN: Very few sources (typically 1), many destination ports
        # Key: src_card very low (near 1), dst_port_card very high (near total_flows)
        if actual_unique_src <= 5 and dst_port_card > total_flows * 0.8:
            confidence = base_confidence + 0.30
            if actual_unique_src == 1:
                confidence += 0.05
            if dst_port_card > total_flows * 0.95:
                confidence += 0.05
            return "Port Scan Attack", min(confidence, 0.98)
        
        # DDoS (SINGLE TARGET): Many sources attacking single or very few destinations
        # This is the most common DDoS pattern - prioritize this check
        # Key: Many unique sources, very few unique destinations (typically 1)
        if actual_unique_dst <= 10 and actual_unique_src > 100 and src_dst_ratio > 10:
            confidence = base_confidence + 0.35
            if actual_unique_dst == 1:  # Single target
                confidence += 0.05
            if actual_unique_src > 1000:  # Large scale
                confidence += 0.03
            if src_dst_ratio > 1000:  # Massive imbalance
                confidence += 0.02
            return "DDoS Attack", min(confidence, 0.98)
        
        # DDoS (HIGH UNIFORMITY): Many diverse sources with even distribution
        # Key: high src_card, low dst_card, high uniformity (even distribution)
        if src_card > 5000 and src_uniformity > 10 and src_dst_ratio > 10:
            confidence = base_confidence + 0.30
            if src_dst_ratio > 100:
                confidence += 0.05
            if src_uniformity > 20:
                confidence += 0.05
            return "DDoS Attack", min(confidence, 0.98)
        
        # Medium DDoS: Similar but with lower thresholds
        if src_card > 1000 and src_uniformity > 5 and src_dst_ratio > 5:
            confidence = base_confidence + 0.25
            if src_uniformity > 10:
                confidence += 0.05
            return "Medium DDoS Attack", confidence
        
        # LEGITIMATE SPIKE: Few sources with power-law distribution
        # Key: low src_card relative to flows, LOW uniformity (skewed), many destinations
        # IMPORTANT: Only classify as legitimate if destinations are diverse (not an attack)
        if (actual_unique_src < total_flows / 50 and 
            src_uniformity < 1.0 and 
            dst_card > 1000 and 
            actual_unique_dst > 100):  # Added: must have many destinations
            confidence = base_confidence + 0.20
            if src_uniformity < 0.7:  # Very skewed
                confidence += 0.05
            if dst_card > total_flows * 0.8:  # Many unique destinations
                confidence += 0.05
            return "Legitimate Spike", confidence
        
        # NORMAL TRAFFIC: Balanced source/destination, moderate uniformity
        # Key: src_card and dst_card similar, uniformity in moderate range
        if actual_unique_src == total_flows and actual_unique_dst == total_flows:
            confidence = base_confidence + 0.15
            if 3 < src_uniformity < 7:  # Reasonable uniformity
                confidence += 0.05
            return "Normal Traffic", confidence
        
        # Fallback for normal-ish traffic
        if src_card < 10000 and 0.5 < src_dst_ratio < 2.0:
            return "Normal Traffic", base_confidence + 0.10
        
        # UNKNOWN/SUSPICIOUS: Doesn't match any clear pattern
        return "Unknown/Suspicious", 0.50

def analyze_flows(flows: List[Flow], cluster_name: str = "Traffic Analysis") -> Dict:
    if not flows:
        return {
            'cluster_name': cluster_name,
            'error': 'No flows to analyze',
            'metrics': {},
            'classification': 'Unknown',
            'confidence': 0.0
        }
    
    cluster = TrafficCluster(cluster_id=cluster_name)
    for flow in flows:
        cluster.add_flow(flow)
    
    metrics = cluster.get_metrics()
    classification, confidence = cluster.classify_traffic(metrics)
    
    return {
        'cluster_name': cluster_name,
        'metrics': metrics,
        'classification': classification,
        'confidence': confidence,
        'cluster': cluster
    }

def print_analysis_results(result: Dict):
    if 'error' in result:
        print(f"\nError: {result['error']}")
        return
    
    metrics = result['metrics']
    classification = result['classification']
    confidence = result['confidence']
    
    print(f"\n{'='*80}")
    print(f"Analysis: {result['cluster_name']}")
    print(f"{'='*80}")
    
    print(f"\nTraffic Metrics:")
    print(f"  Total Flows: {metrics['total_flows']:,}")
    
    print(f"\n  Source IP Analysis:")
    print(f"    Actual Unique: {metrics['actual_unique_src_ips']:,}")
    print(f"    Estimated Cardinality: {metrics['src_ip_cardinality']:,.0f}")
    print(f"    Entropy: {metrics['src_ip_entropy']:.2f} bits")
    print(f"    Uniformity Ratio: {metrics['src_ip_uniformity']:.3f}")
    
    print(f"\n  Destination IP Analysis:")
    print(f"    Actual Unique: {metrics['actual_unique_dst_ips']:,}")
    print(f"    Estimated Cardinality: {metrics['dst_ip_cardinality']:,.0f}")
    print(f"    Entropy: {metrics['dst_ip_entropy']:.2f} bits")
    print(f"    Uniformity Ratio: {metrics['dst_ip_uniformity']:.3f}")
    
    print(f"\n  Port Analysis:")
    print(f"    Src Port Entropy: {metrics['src_port_entropy']:.2f} bits")
    print(f"    Dst Port Entropy: {metrics['dst_port_entropy']:.2f} bits")
    print(f"    Dst Port Cardinality: {metrics['dst_port_cardinality']:,.0f}")
    
    print(f"\n  Derived Metrics:")
    print(f"    Src/Dst Ratio: {metrics['src_dst_ratio']:.2f}")
    
    print(f"\nClassification: {classification}")
    print(f"   Confidence: {confidence*100:.1f}%")

def analyze_file(filepath: str, cluster_name: str = None) -> Dict:
    import os
    
    if cluster_name is None:
        cluster_name = os.path.basename(filepath)
    
    print(f"\n{'='*80}")
    print(f"Reading flows from: {filepath}")
    print(f"{'='*80}")
    
    try:
        flows = parse_flows(filepath)
        print(f"Successfully read {len(flows):,} flows")
        
        if not flows:
            print("Warning: No valid flows found in file")
            return {
                'cluster_name': cluster_name,
                'error': 'No valid flows found',
                'metrics': {},
                'classification': 'Unknown',
                'confidence': 0.0
            }
        
        # Analyze the flows
        result = analyze_flows(flows, cluster_name)
        print_analysis_results(result)
        
        return result
        
    except FileNotFoundError:
        error_msg = f"File not found: {filepath}"
        print(f"\nError: {error_msg}")
        return {
            'cluster_name': cluster_name,
            'error': error_msg,
            'metrics': {},
            'classification': 'Unknown',
            'confidence': 0.0
        }
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        print(f"\nError: {error_msg}")
        return {
            'cluster_name': cluster_name,
            'error': error_msg,
            'metrics': {},
            'classification': 'Unknown',
            'confidence': 0.0
        }

def run_simulation():
    print("=" * 80)
    print("NETWORK TRAFFIC SIMULATION - ENTROPY & CARDINALITY ANALYSIS")
    print("=" * 80)
    
    scenarios = [
        ("DDoS Attack (Large)", "ddos", 50000),
        ("Port Scan", "portscan", 10000),
        ("Legitimate Spike", "legitimate_spike", 30000),
        ("Normal Traffic", "normal", 5000),
    ]
    
    all_clusters = []
    
    for scenario_name, attack_type, num_flows in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name} ({num_flows} flows)")
        print(f"{'='*80}")
        
        random.seed(42)
        flows = generate_attack_flows(num_flows, attack_type)
        
        result = analyze_flows(flows, scenario_name)
        all_clusters.append(result)
        
        print_analysis_results(result)
    
    print(f"\n{'='*80}")
    print("RANKING: CLUSTERS BY SOURCE IP ENTROPY")
    print(f"{'='*80}")
    
    sorted_by_entropy = sorted(all_clusters, key=lambda x: x['metrics']['src_ip_entropy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Cluster':<30} {'Entropy':<12} {'Cardinality':<15} {'Classification':<25}")
    print("-" * 100)
    
    for rank, item in enumerate(sorted_by_entropy, 1):
        metrics = item['metrics']
        print(f"{rank:<6} {metrics['cluster_id']:<30} "
              f"{metrics['src_ip_entropy']:>10.2f}  "
              f"{metrics['src_ip_cardinality']:>13,.0f}  "
              f"{item['classification']:<25}")
    
    print(f"\n{'='*80}")
    print("RANKING: CLUSTERS BY UNIFORMITY (Attack Indicator)")
    print(f"{'='*80}")
    
    sorted_by_uniformity = sorted(all_clusters, key=lambda x: x['metrics']['src_ip_uniformity'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Cluster':<30} {'Uniformity':<12} {'Entropy':<10} {'Classification':<25}")
    print("-" * 100)
    
    for rank, item in enumerate(sorted_by_uniformity, 1):
        metrics = item['metrics']
        print(f"{rank:<6} {metrics['cluster_id']:<30} "
              f"{metrics['src_ip_uniformity']:>10.3f}  "
              f"{metrics['src_ip_entropy']:>8.2f}  "
              f"{item['classification']:<25}")
    
    print(f"\n{'='*80}")
    print("ANOMALY DETECTION SUMMARY")
    print(f"{'='*80}")
    
    attacks_detected = [item for item in all_clusters if "Attack" in item['classification']]
    legitimate = [item for item in all_clusters if "Attack" not in item['classification']]
    
    print(f"\nAttacks Detected: {len(attacks_detected)}")
    for item in attacks_detected:
        print(f"  - {item['metrics']['cluster_id']}: {item['classification']} "
              f"(Confidence: {item['confidence']*100:.1f}%)")
    
    print(f"\nLegitimate Traffic: {len(legitimate)}")
    for item in legitimate:
        print(f"  - {item['metrics']['cluster_id']}: {item['classification']} "
              f"(Confidence: {item['confidence']*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("DETECTION PERFORMANCE")
    print(f"{'='*80}")
    
    total = len(all_clusters)
    
    def matches(expected, detected):
        if "DDoS" in expected and "DDoS" in detected:
            return True
        if "Port Scan" in expected and "Port Scan" in detected:
            return True
        if "Legitimate" in expected and "Legitimate" in detected:
            return True
        if "Normal" in expected and "Normal" in detected:
            return True
        return False
    
    correct_classifications = sum(1 for item in all_clusters 
                                  if matches(item['cluster'].cluster_id, item['classification']))
    
    accuracy = (correct_classifications / total) * 100 if total > 0 else 0
    print(f"\nClassification Accuracy: {accuracy:.1f}% ({correct_classifications}/{total})")
    
    avg_confidence = sum(item['confidence'] for item in all_clusters) / len(all_clusters)
    print(f"Average Confidence: {avg_confidence*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("DETAILED CLASSIFICATION BREAKDOWN")
    print(f"{'='*80}")
    print(f"\n{'Expected':<30} {'Detected':<30} {'Match':<10} {'Confidence':<12}")
    print("-" * 82)
    for item in all_clusters:
        expected = item['cluster'].cluster_id
        detected = item['classification']
        is_match = matches(expected, detected)
        match = "✓" if is_match else "✗"
        print(f"{expected:<30} {detected:<30} {match:<10} {item['confidence']*100:>10.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        cluster_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        print("=" * 80)
        print("NETWORK TRAFFIC ANALYSIS - FILE MODE")
        print("=" * 80)
        
        result = analyze_file(filepath, cluster_name)
        
        if 'error' not in result:
            print(f"\n{'='*80}")
            print("ANALYSIS SUMMARY")
            print(f"{'='*80}")
            print(f"\nFile: {filepath}")
            print(f"Flows Analyzed: {result['metrics']['total_flows']:,}")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
    else:
        run_simulation()