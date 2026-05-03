from typing import Tuple, Dict
from dataclasses import dataclass, field
from typing import List
import zlib
import hashlib
import random
import math
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
        # DDoS: Many sources, single target
        target_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        
        for i in range(num_flows):
            src_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = "00080"
            protocol = "06"
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    elif attack_type == "portscan":
        # Port scan: Single source, many ports
        attacker_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        target_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        
        for i in range(num_flows):
            src_ip = attacker_ip
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = f"{random.randint(1, 65535):05d}"
            protocol = "06"
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    elif attack_type == "traffic_spike":
        # Traffic Spike: Sudden surge of legitimate-looking traffic from many sources
        popular_services = [80, 443, 8080, 8443, 3000]
        
        for i in range(num_flows):
            src_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
            dst_ip = f"{random.randint(1, 10):03d}{random.randint(1, 10):03d}{random.randint(1, 10):03d}{random.randint(1, 10):03d}"
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = f"{random.choice(popular_services):05d}"
            protocol = "06"
            flows.append(Flow(src_ip, dst_ip, src_port, dst_port, protocol))
    
    elif attack_type == "slowloris":
        # Slowloris: Few sources, many slow connections to same target
        # Low source diversity, single target, many connections per source
        target_ip = f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}"
        num_attackers = max(1, num_flows // 100)  # Few attackers, many connections each
        attacker_ips = [f"{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}{random.randint(1, 255):03d}" 
                       for _ in range(num_attackers)]
        
        for i in range(num_flows):
            src_ip = random.choice(attacker_ips)  # Reuse attacker IPs
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = "00080"
            protocol = "06"
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

class MLTrafficClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_map = {
            0: 'Normal',
            1: 'DDoS',
            2: 'Port Scan',
            3: 'Traffic Spike',
            4: 'Slowloris'
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.is_trained = False
        
    def extract_features(self, metrics: Dict) -> np.ndarray:
        features = [
            metrics['total_flows'],
            metrics['actual_unique_src_ips'],
            metrics['actual_unique_dst_ips'],
            metrics['repetition_ratio'],
            metrics['src_ip_entropy'],
            metrics['dst_ip_entropy'],
            metrics['src_port_entropy'],
            metrics['dst_port_entropy'],
            metrics['src_ip_cardinality'],
            metrics['dst_ip_cardinality'],
            metrics['src_port_cardinality'],
            metrics['dst_port_cardinality'],
            metrics['src_ip_uniformity'],
            metrics['dst_ip_uniformity'],
            metrics['src_dst_ratio'],
        ]
        return np.array(features).reshape(1, -1)
    
    def generate_training_data(self, samples_per_class: int = 200):
        X = []
        y = []
        
        print("Generating training data...")
        
        # Normal traffic - varied patterns
        for i in range(samples_per_class):
            num_flows = random.randint(100, 5000)
            flows = generate_attack_flows(num_flows, "normal")
            cluster = TrafficCluster(f"normal_{i}")
            for flow in flows:
                cluster.add_flow(flow)
            metrics = cluster.get_metrics()
            X.append(self.extract_features(metrics).flatten())
            y.append(self.reverse_label_map['Normal'])
        
        # DDoS attacks - varied scales
        for i in range(samples_per_class):
            num_flows = random.randint(1000, 50000)
            flows = generate_attack_flows(num_flows, "ddos")
            cluster = TrafficCluster(f"ddos_{i}")
            for flow in flows:
                cluster.add_flow(flow)
            metrics = cluster.get_metrics()
            X.append(self.extract_features(metrics).flatten())
            y.append(self.reverse_label_map['DDoS'])
        
        # Port scans - varied intensities
        for i in range(samples_per_class):
            num_flows = random.randint(500, 10000)
            flows = generate_attack_flows(num_flows, "portscan")
            cluster = TrafficCluster(f"portscan_{i}")
            for flow in flows:
                cluster.add_flow(flow)
            metrics = cluster.get_metrics()
            X.append(self.extract_features(metrics).flatten())
            y.append(self.reverse_label_map['Port Scan'])
        
        # Traffic Spike - sudden surge of legitimate traffic
        for i in range(samples_per_class):
            num_flows = random.randint(5000, 100000)
            flows = generate_attack_flows(num_flows, "traffic_spike")
            cluster = TrafficCluster(f"traffic_spike_{i}")
            for flow in flows:
                cluster.add_flow(flow)
            metrics = cluster.get_metrics()
            X.append(self.extract_features(metrics).flatten())
            y.append(self.reverse_label_map['Traffic Spike'])
        
        # Slowloris - slow HTTP DoS attack
        for i in range(samples_per_class):
            num_flows = random.randint(1000, 20000)
            flows = generate_attack_flows(num_flows, "slowloris")
            cluster = TrafficCluster(f"slowloris_{i}")
            for flow in flows:
                cluster.add_flow(flow)
            metrics = cluster.get_metrics()
            X.append(self.extract_features(metrics).flatten())
            y.append(self.reverse_label_map['Slowloris'])
        
        return np.array(X), np.array(y)
    
    def train(self, samples_per_class: int = 200):
        X, y = self.generate_training_data(samples_per_class)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score*100:.2f}%")
        print(f"Testing accuracy: {test_score*100:.2f}%")
        
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.label_map.values())))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        self.is_trained = True
        
        feature_names = [
            'total_flows', 'unique_src_ips', 'unique_dst_ips', 'repetition_ratio',
            'src_ip_entropy', 'dst_ip_entropy', 'src_port_entropy', 'dst_port_entropy',
            'src_ip_cardinality', 'dst_ip_cardinality', 'src_port_cardinality', 
            'dst_port_cardinality', 'src_ip_uniformity', 'dst_ip_uniformity', 'src_dst_ratio'
        ]
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Important Features:")
        for fname, importance in feature_importance[:10]:
            print(f"  {fname}: {importance:.4f}")
    
    def predict(self, metrics: Dict) -> Tuple[str, float]:
        if not self.is_trained:
            return "Unknown (Model not trained)", 0.0
        
        features = self.extract_features(metrics)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        classification = self.label_map[prediction]
        confidence = probabilities[prediction]
        
        return classification, confidence
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

def analyze_flows(flows: List[Flow], cluster_name: str = "Traffic Analysis", 
                 classifier: MLTrafficClassifier = None) -> Dict:
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
    
    if classifier and classifier.is_trained:
        classification, confidence = classifier.predict(metrics)
    else:
        if metrics['actual_unique_dst_ips'] <= 5 and metrics['src_dst_ratio'] > 10:
            classification = "DDoS"
            confidence = 0.7
        elif metrics['actual_unique_src_ips'] <= 5 and metrics['dst_port_cardinality'] > metrics['total_flows'] * 0.7:
            classification = "Port Scan"
            confidence = 0.7
        else:
            classification = "Normal"
            confidence = 0.6
    
    return {
        'cluster_name': cluster_name,
        'metrics': metrics,
        'classification': classification,
        'confidence': confidence,
        'cluster': cluster
    }

def print_analysis_results(result: Dict):
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    metrics = result['metrics']
    classification = result['classification']
    confidence = result['confidence']
    
    print(f"\n{'='*80}")
    print(f"Analysis: {result['cluster_name']}")
    print(f"{'='*80}")
    
    print(f"\nTraffic Metrics:")
    print(f"  Total Flows: {metrics['total_flows']:,}")
    print(f"  Unique Flows: {metrics['unique_flows']:,}")
    print(f"  Repetition Ratio: {metrics['repetition_ratio']:.2f}")
    
    print(f"\nSource IP Analysis:")
    print(f"  Actual Unique: {metrics['actual_unique_src_ips']:,}")
    print(f"  Estimated Cardinality: {metrics['src_ip_cardinality']:,.0f}")
    print(f"  Entropy: {metrics['src_ip_entropy']:.2f} bits")
    print(f"  Uniformity Ratio: {metrics['src_ip_uniformity']:.3f}")
    
    print(f"\nDestination IP Analysis:")
    print(f"  Actual Unique: {metrics['actual_unique_dst_ips']:,}")
    print(f"  Estimated Cardinality: {metrics['dst_ip_cardinality']:,.0f}")
    print(f"  Entropy: {metrics['dst_ip_entropy']:.2f} bits")
    print(f"  Uniformity Ratio: {metrics['dst_ip_uniformity']:.3f}")
    
    print(f"\nPort Analysis:")
    print(f"  Src Port Entropy: {metrics['src_port_entropy']:.2f} bits")
    print(f"  Dst Port Entropy: {metrics['dst_port_entropy']:.2f} bits")
    print(f"  Dst Port Cardinality: {metrics['dst_port_cardinality']:,.0f}")
    
    print(f"\nDerived Metrics:")
    print(f"  Src/Dst Ratio: {metrics['src_dst_ratio']:.2f}")
    
    print(f"\nClassification: {classification}")
    print(f"  Confidence: {confidence*100:.1f}%")

def analyze_file(filepath: str, cluster_name: str = None, 
                classifier: MLTrafficClassifier = None) -> Dict:
    import os
    
    if cluster_name is None:
        cluster_name = os.path.basename(filepath)
    
    print(f"\n{'='*80}")
    print(f"Reading flows from: {filepath}")
    print(f"{'='*80}")
    
    try:
        flows = parse_flows(filepath)
        print(f"✓ Successfully read {len(flows):,} flows")
        
        if not flows:
            print("No valid flows found in file")
            return {
                'cluster_name': cluster_name,
                'error': 'No valid flows found',
                'metrics': {},
                'classification': 'Unknown',
                'confidence': 0.0
            }
        
        result = analyze_flows(flows, cluster_name, classifier)
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
        print(f"\n Error: {error_msg}")
        return {
            'cluster_name': cluster_name,
            'error': error_msg,
            'metrics': {},
            'classification': 'Unknown',
            'confidence': 0.0
        }

def run_simulation_with_ml():
    print("=" * 80)
    print("NETWORK TRAFFIC ANALYSIS - ML-BASED CLASSIFICATION")
    print("=" * 80)
    
    classifier = MLTrafficClassifier()
    classifier.train(samples_per_class=300)
    
    model_path = "traffic_classifier.pkl"
    classifier.save_model(model_path)
    
    print(f"\n{'='*80}")
    print("TESTING ON NEW SCENARIOS")
    print(f"{'='*80}")
    
    scenarios = [
        ("Normal", "normal", 3000),
        ("DDoS", "ddos", 20000),
        ("Port Scan", "portscan", 8000),
        ("Traffic Spike", "traffic_spike", 10000),
        ("Slowloris", "slowloris", 10000)
    ]
    
    results = []
    
    for scenario_name, attack_type, num_flows in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*80}")
        
        random.seed(42)
        flows = generate_attack_flows(num_flows, attack_type)
        
        result = analyze_flows(flows, scenario_name, classifier)
        results.append(result)
        
        print_analysis_results(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Expected':<25} {'Predicted':<25} {'Confidence':<12} {'Match'}")
    print("-" * 75)
    
    for i, (name, _, _) in enumerate(scenarios):
        result = results[i]
        expected = name.split()[0]  # Get first word (Normal, DDoS, Port)
        predicted = result['classification']
        confidence = result['confidence']
        match = "✓" if expected.lower() in predicted.lower() else "✗"
        
        print(f"{name:<25} {predicted:<25} {confidence*100:>10.1f}%  {match}")

if __name__ == "__main__":
    import sys
    
    model_path = "traffic_classifier.pkl"
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        cluster_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        print("=" * 80)
        print("NETWORK TRAFFIC ANALYSIS - FILE MODE (ML)")
        print("=" * 80)
        
        classifier = MLTrafficClassifier()
        if os.path.exists(model_path):
            print("\nLoading pre-trained model...")
            classifier.load_model(model_path)
        else:
            print("\nNo pre-trained model found. Training new model...")
            classifier.train(samples_per_class=300)
            classifier.save_model(model_path)
        
        result = analyze_file(filepath, cluster_name, classifier)
        
        if 'error' not in result:
            print(f"\n{'='*80}")
            print("ANALYSIS SUMMARY")
            print(f"{'='*80}")
            print(f"\n✓ File: {filepath}")
            print(f"Flows Analyzed: {result['metrics']['total_flows']:,}")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
    else:
        run_simulation_with_ml()