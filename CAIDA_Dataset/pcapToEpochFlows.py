#!/usr/bin/env python3
import sys
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path


def ip_to_12_digits(ip_str):
    """Convert IP to 12 digit format"""
    octets = ip_str.split('.')
    return ''.join(f'{int(octet):03d}' for octet in octets)


def process_tshark_line(line, first_timestamp_ms, epoch_ms):
    """Process a single tshark output line"""
    try:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            return None, None
        
        timestamp, src_ip, dst_ip, protocol, tcp_sport, tcp_dport, udp_sport, udp_dport = parts[:8]
        
        # Skip if no IP addresses
        if not src_ip or not dst_ip:
            return None, None
        
        # Calculate epoch
        timestamp_ms = float(timestamp) * 1000
        relative_time = timestamp_ms - first_timestamp_ms
        epoch_num = int(relative_time // epoch_ms)
        
        # Format flow
        src_ip_str = ip_to_12_digits(src_ip)
        dst_ip_str = ip_to_12_digits(dst_ip)
        
        # Get ports (TCP or UDP)
        src_port = 0
        dst_port = 0
        if tcp_sport:
            src_port = int(tcp_sport)
            dst_port = int(tcp_dport) if tcp_dport else 0
        elif udp_sport:
            src_port = int(udp_sport)
            dst_port = int(udp_dport) if udp_dport else 0
        
        src_port_str = f"{src_port:05d}"
        dst_port_str = f"{dst_port:05d}"
        
        # Protocol is already a number from ip.proto
        proto_num = int(protocol) if protocol else 0
        protocol_str = f"{proto_num:02d}"
        
        flow_str = src_ip_str + dst_ip_str + src_port_str + dst_port_str + protocol_str
        
        return epoch_num, flow_str
    except Exception as e:
        return None, None


def process_chunk_worker(chunk_file, first_timestamp_ms, epoch_ms, worker_id):
    """Process a chunk file and return epoch flows"""
    print(f"[Worker-{worker_id}] Processing {chunk_file}")
    
    local_epoch_flows = defaultdict(set)
    line_count = 0
    flow_count = 0
    
    try:
        with open(chunk_file, 'r') as f:
            for line in f:
                line_count += 1
                epoch_num, flow_str = process_tshark_line(line, first_timestamp_ms, epoch_ms)
                
                if epoch_num is not None and flow_str is not None:
                    if flow_str not in local_epoch_flows[epoch_num]:
                        local_epoch_flows[epoch_num].add(flow_str)
                        flow_count += 1
                
                if line_count % 100000 == 0:
                    print(f"[Worker-{worker_id}] Processed {line_count} lines, {flow_count} flows")
        
        print(f"[Worker-{worker_id}] Completed! {line_count} lines, {flow_count} flows, {len(local_epoch_flows)} epochs")
        return local_epoch_flows
        
    except Exception as e:
        print(f"[Worker-{worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return defaultdict(set)


def extract_with_tshark(pcap_file, output_file):
    """Extract packet info using tshark - MUCH faster than scapy"""
    print("Extracting packets using tshark (this is much faster than scapy)...")
    print("This may take a few minutes for 27 million packets...")
    
    # tshark command to extract: timestamp, src_ip, dst_ip, protocol, src_port, dst_port
    cmd = [
        'tshark',
        '-r', pcap_file,
        '-T', 'fields',
        '-e', 'frame.time_epoch',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'ip.proto',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'udp.srcport',
        '-e', 'udp.dstport',
        '-E', 'separator=\t',
        '-E', 'occurrence=f'
    ]
    
    try:
        # Run tshark and save to file
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            
            # Monitor progress
            while process.poll() is None:
                pass
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                print(f"tshark error: {stderr}")
                return False
        
        print(f"Extraction complete! Output saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print("ERROR: tshark not found. Please install Wireshark/tshark:")
        print("  Ubuntu/Debian: sudo apt-get install tshark")
        print("  CentOS/RHEL: sudo yum install wireshark")
        print("  macOS: brew install wireshark")
        return False
    except Exception as e:
        print(f"Error running tshark: {e}")
        return False


def split_file(input_file, num_chunks):
    """Split the extracted data into chunks for parallel processing"""
    print(f"\nSplitting data into {num_chunks} chunks for parallel processing...")
    
    # Count total lines
    print("Counting lines...")
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines:,}")
    
    lines_per_chunk = total_lines // num_chunks + 1
    chunk_files = []
    
    with open(input_file, 'r') as f:
        chunk_num = 0
        chunk_file = f"{input_file}.chunk_{chunk_num}"
        chunk_files.append(chunk_file)
        chunk_out = open(chunk_file, 'w')
        
        for i, line in enumerate(f, 1):
            chunk_out.write(line)
            
            if i % lines_per_chunk == 0 and chunk_num < num_chunks - 1:
                chunk_out.close()
                chunk_num += 1
                chunk_file = f"{input_file}.chunk_{chunk_num}"
                chunk_files.append(chunk_file)
                chunk_out = open(chunk_file, 'w')
                print(f"  Created chunk {chunk_num}/{num_chunks}")
        
        chunk_out.close()
    
    print(f"Created {len(chunk_files)} chunk files")
    return chunk_files


def get_first_timestamp(extracted_file):
    """Get the first timestamp from extracted data"""
    print("\nGetting first timestamp...")
    with open(extracted_file, 'r') as f:
        first_line = f.readline()
        timestamp = float(first_line.split('\t')[0])
        print(f"First timestamp: {timestamp}")
        return timestamp * 1000  # Convert to milliseconds


def process_pcap_ultra_fast(pcap_file, epoch_ms=280, output_dir='flows', num_workers=8):
    """Ultra-fast processing using tshark + parallel processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ULTRA-FAST PCAP FLOW PROCESSOR")
    print("=" * 80)
    print(f"PCAP file: {pcap_file}")
    print(f"Epoch size: {epoch_ms}ms")
    print(f"Workers: {num_workers}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Step 1: Extract using tshark
    extracted_file = f"{pcap_file}.extracted.txt"
    
    if not os.path.exists(extracted_file):
        print("\nStep 1/4: Extracting packets with tshark...")
        if not extract_with_tshark(pcap_file, extracted_file):
            return
    else:
        print(f"\nStep 1/4: Using existing extracted file: {extracted_file}")
    
    # Step 2: Get first timestamp
    print("\nStep 2/4: Getting first timestamp...")
    first_timestamp_ms = get_first_timestamp(extracted_file)
    
    # Step 3: Split into chunks
    print("\nStep 3/4: Splitting into chunks...")
    chunk_files = split_file(extracted_file, num_workers)
    
    # Step 4: Process chunks in parallel
    print("\nStep 4/4: Processing chunks in parallel...")
    print("=" * 80)
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_chunk_worker,
                chunk_file,
                first_timestamp_ms,
                epoch_ms,
                worker_id + 1
            ): worker_id
            for worker_id, chunk_file in enumerate(chunk_files)
        }
        
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error from worker {worker_id}: {e}")
    
    # Merge results
    print("\n" + "=" * 80)
    print("Merging results from all workers...")
    merged = defaultdict(set)
    
    for local_flows in all_results:
        for epoch_num, flows in local_flows.items():
            merged[epoch_num].update(flows)
    
    total_flows = sum(len(flows) for flows in merged.values())
    print(f"Total unique flows: {total_flows:,}")
    print(f"Total epochs: {len(merged):,}")
    
    # Write flow files
    print("\n" + "=" * 80)
    print(f"Writing flow files to {output_dir}/")
    print("=" * 80)
    
    for epoch_num in sorted(merged.keys()):
        flows = merged[epoch_num]
        filename = os.path.join(output_dir, f'epoch_{epoch_num:06d}.txt')
        
        with open(filename, 'w') as f:
            for flow in sorted(flows):
                f.write(flow + '\n')
        
        if epoch_num % 10 == 0 or len(flows) > 1000:
            print(f"  Epoch {epoch_num:6d}: {len(flows):8,d} flows -> {filename}")
    
    # Cleanup chunk files
    print("\n" + "=" * 80)
    print("Cleaning up temporary chunk files...")
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Created {len(merged):,} epoch files")
    print(f"Total unique flows: {total_flows:,}")
    print(f"Output directory: {output_dir}/")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python pcapToEpochFlows.py <pcap_file> [epoch_ms] [output_dir] [num_workers]")
        print("\nArguments:")
        print("  pcap_file   : Path to the PCAP file")
        print("  epoch_ms    : Epoch size in milliseconds (default: 280)")
        print("  output_dir  : Output directory for flow files (default: 'flows')")
        print("  num_workers : Number of parallel workers (default: 8)")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    epoch_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 280
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'flows'
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    
    if not os.path.exists(pcap_file):
        print(f"Error: PCAP file not found: {pcap_file}")
        sys.exit(1)
    
    try:
        process_pcap_ultra_fast(pcap_file, epoch_ms, output_dir, num_workers)
    except Exception as e:
        print(f"Error processing PCAP file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
