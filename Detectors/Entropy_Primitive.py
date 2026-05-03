from typing import Tuple
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
        
    def calculate_entropy(self, cms: CountMinSketch, items: List[str]) -> float:
        if cms.total_count == 0:
            return 0.0
        
        entropy = 0.0
        non_zero_buckets = 0
        
        for item in items:
            freq = cms.estimate(item)
            if freq > 0:
                non_zero_buckets += 1
                prob = freq / cms.total_count
                index = int(prob * 256)
                if index >= 256:
                    index = 255
                if index > 0:
                    entropy -= prob * self.log_table[index]
            
        return entropy
    
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
    
    def get_src_ip_entropy(self, unique_src_ips: List[str]) -> float:
        return self.calculate_entropy(self.src_ip_cms, unique_src_ips)
    
    def get_dst_ip_entropy(self, unique_dst_ips: List[str]) -> float:
        return self.calculate_entropy(self.dst_ip_cms, unique_dst_ips)
    
    def get_src_port_entropy(self, unique_src_ports: List[str]) -> float:
        return self.calculate_entropy(self.src_port_cms, unique_src_ports)
    
    def get_dst_port_entropy(self, unique_dst_ports: List[str]) -> float:
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