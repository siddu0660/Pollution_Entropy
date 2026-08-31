import hashlib
import random
import csv
import statistics
import argparse
import math
import concurrent.futures
from typing import List, Tuple, Dict
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable


@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    
    def get_signature(self) -> int:
        sig_str = f"{self.src_ip}:{self.dst_ip}:{self.src_port}:{self.dst_port}:{self.protocol}"
        return int(hashlib.sha256(sig_str.encode()).hexdigest(), 16)

class IBLT:
    def __init__(self, num_cells: int, num_hash: int = 3):
        self.num_cells = max(1, int(num_cells))
        self.num_hash = num_hash
        self.xor_sum = [0] * self.num_cells
        self.count = [0] * self.num_cells
    
    def _hash(self, signature: int, hash_index: int) -> int:
        return (signature + hash_index * 0x5bd1e995) % self.num_cells
    
    def insert(self, signature: int):
        for i in range(self.num_hash):
            cell_idx = self._hash(signature, i)
            self.xor_sum[cell_idx] ^= signature
            self.count[cell_idx] += 1
    
    def subtract(self, other: 'IBLT') -> 'IBLT':
        result = IBLT(self.num_cells, self.num_hash)
        for i in range(self.num_cells):
            result.xor_sum[i] = self.xor_sum[i] ^ other.xor_sum[i]
            result.count[i] = self.count[i] - other.count[i]
        return result
    
    def decode(self) -> Tuple[List[int], bool]:
        decoded = []
        decoded_set = set()
        working_xor = self.xor_sum.copy()
        working_count = self.count.copy()
        pure_cells = [i for i in range(self.num_cells) if abs(working_count[i]) == 1]
        
        while pure_cells:
            i = pure_cells.pop(0)
            if abs(working_count[i]) != 1: continue
            signature = working_xor[i]
            if signature in decoded_set:
                working_count[i] = 0; working_xor[i] = 0
                continue
            decoded.append(signature)
            decoded_set.add(signature)
            sign = 1 if working_count[i] > 0 else -1
            for j in range(self.num_hash):
                cell_idx = self._hash(signature, j)
                working_xor[cell_idx] ^= signature
                working_count[cell_idx] -= sign
                if abs(working_count[cell_idx]) == 1:
                    pure_cells.append(cell_idx)
        return decoded, all(c == 0 for c in working_count)

def run_single_simulation(iblt_size, num_hash, flows, num_benign, num_malicious, loss_rate):
    total_needed = num_benign + num_malicious
    if len(flows) < total_needed:
         flows = flows * (math.ceil(total_needed / len(flows)))

    selected_flows = random.sample(flows, total_needed)
    benign_flows = selected_flows[:num_benign]
    malicious_flows = selected_flows[num_benign:]
    
    num_lost = max(1, int(num_benign * loss_rate))
    lost_flows_subset = benign_flows[:num_lost]
    lost_signatures = {f.get_signature() for f in lost_flows_subset}
    
    upstream_iblt = IBLT(iblt_size, num_hash)
    downstream_iblt = IBLT(iblt_size, num_hash)
    
    for flow in benign_flows:
        sig = flow.get_signature()
        upstream_iblt.insert(sig)
        if sig not in lost_signatures:
            downstream_iblt.insert(sig)
    for flow in malicious_flows:
        upstream_iblt.insert(flow.get_signature())
            
    diff_iblt = upstream_iblt.subtract(downstream_iblt)
    decoded_sigs, _ = diff_iblt.decode()
    
    decoded_benign_count = sum(1 for sig in decoded_sigs if sig in lost_signatures)
    undecodable_count = len(lost_signatures) - decoded_benign_count
    return (undecodable_count / len(lost_signatures)) * 100.0

def worker_task(args):
    (mal_pct, loss_pct, num_runs, capacity, iblt_size, num_hash, flows) = args
    num_malicious = int(capacity * mal_pct)
    num_benign = capacity - num_malicious
    
    run_results = []
    for _ in range(num_runs):
        res = run_single_simulation(iblt_size, num_hash, flows, num_benign, num_malicious, loss_pct)
        run_results.append(res)
    return mal_pct, loss_pct, statistics.mean(run_results)

def generate_synthetic_flows(num_flows: int) -> List[Flow]:
    return [Flow(f"10.{i%255}.{i//255%255}.1", "10.1.0.1", i%65535, 80, 6) for i in range(num_flows)]

def export_matrix_csv(matrix, loss_rates, mal_rates, capacity, design_rate, iblt_size, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["Metadata", "Value"])
        writer.writerow(["Capacity", capacity])
        writer.writerow(["Design Rate", design_rate])
        writer.writerow(["IBLT Size", iblt_size])
        writer.writerow([])
        
        header = ["Malicious% \ Loss%"] + [f"{lr*100:.2f}%" for lr in loss_rates]
        writer.writerow(header)
        
        for mal in mal_rates:
            row = [f"{mal*100:.2f}%"]
            for loss in loss_rates:
                val = matrix.get(mal, {}).get(loss, 0.0)
                row.append(f"{val:.4f}")
            writer.writerow(row)
            
    print(f"Analysis saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="IBLT Attack Simulation")
    parser.add_argument('-c', '--capacity', type=int, default=10000, 
                        help='Total flow capacity (default: 10000)')
    parser.add_argument('-r', '--rate', type=float, default=0.02, 
                        help='Design Loss Rate (e.g. 0.02 for 2%) (default: 0.02)')
    args = parser.parse_args()

    CAPACITY = args.capacity        
    DESIGN_LOSS_RATE = args.rate 
    OVERHEAD = 1.3
    IBLT_SIZE = int(CAPACITY * DESIGN_LOSS_RATE * OVERHEAD)
    
    NUM_HASH = 3
    NUM_RUNS = 30
    
    LOSS_PERCENTAGES = [0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.0185, 0.02, 0.0225, 0.025, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
    MALICIOUS_PERCENTAGES = [0.0000, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01, 0.013, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

    print("="*60)
    print(f"PARALLEL IBLT SIMULATION")
    print(f"Config: Capacity={CAPACITY}, DesignRate={DESIGN_LOSS_RATE}")
    print(f"IBLT Size: {IBLT_SIZE} cells")
    print("="*60)

    flows = generate_synthetic_flows(CAPACITY * 2)
    tasks = []
    for mal_pct in MALICIOUS_PERCENTAGES:
        for loss_pct in LOSS_PERCENTAGES:
            tasks.append((mal_pct, loss_pct, NUM_RUNS, CAPACITY, IBLT_SIZE, NUM_HASH, flows))
            
    results_matrix = {mal: {} for mal in MALICIOUS_PERCENTAGES}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_task, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), unit="cell"):
            mal_pct, loss_pct, result = future.result()
            results_matrix[mal_pct][loss_pct] = result

    output_filename = f"matrix_{CAPACITY}_{DESIGN_LOSS_RATE}.csv"
    
    export_matrix_csv(results_matrix, LOSS_PERCENTAGES, MALICIOUS_PERCENTAGES, 
                      CAPACITY, DESIGN_LOSS_RATE, IBLT_SIZE, output_filename)

if __name__ == "__main__":
    main()