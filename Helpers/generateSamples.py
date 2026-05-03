#!/usr/bin/env python3
"""
Generate sample network flow files for testing.

Flow format (36 characters per line):
- Characters 0-11:  Source IP (12 digits, e.g., 192168001001 = 192.168.1.1)
- Characters 12-23: Destination IP (12 digits)
- Characters 24-28: Source Port (5 digits)
- Characters 29-33: Destination Port (5 digits)
- Characters 34-35: Protocol (2 digits, 06=TCP, 17=UDP)
"""

import random

def generate_sample_ddos_file(filename: str = "sample_ddos.txt", num_flows: int = 1000):
    """Generate a sample DDoS attack flow file."""
    print(f"Generating DDoS attack sample: {filename} ({num_flows} flows)")
    
    target_ip = "192168001010"  # 192.168.1.10
    
    with open(filename, 'w') as f:
        for i in range(num_flows):
            # Random source IPs (many different attackers)
            src_ip = f"{random.randint(1, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(1, 254):03d}"
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = "00080"  # Port 80
            protocol = "06"  # TCP
            
            line = f"{src_ip}{dst_ip}{src_port}{dst_port}{protocol}\n"
            f.write(line)
    
    print(f"✓ Created {filename}")

def generate_sample_portscan_file(filename: str = "sample_portscan.txt", num_flows: int = 1000):
    """Generate a sample port scan flow file."""
    print(f"Generating port scan sample: {filename} ({num_flows} flows)")
    
    attacker_ip = "010020030040"  # 1.2.3.4
    target_ip = "192168001010"  # 192.168.1.10
    
    with open(filename, 'w') as f:
        for i in range(num_flows):
            src_ip = attacker_ip
            dst_ip = target_ip
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = f"{random.randint(1, 65535):05d}"  # Scanning all ports
            protocol = "06"  # TCP
            
            line = f"{src_ip}{dst_ip}{src_port}{dst_port}{protocol}\n"
            f.write(line)
    
    print(f"✓ Created {filename}")

def generate_sample_normal_file(filename: str = "sample_normal.txt", num_flows: int = 1000):
    """Generate a sample normal traffic flow file."""
    print(f"Generating normal traffic sample: {filename} ({num_flows} flows)")
    
    with open(filename, 'w') as f:
        for i in range(num_flows):
            # Random source and destination IPs
            src_ip = f"{random.randint(1, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(1, 254):03d}"
            dst_ip = f"{random.randint(1, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(1, 254):03d}"
            src_port = f"{random.randint(1024, 65535):05d}"
            
            # Common destination ports
            dst_port_val = random.choice([80, 443, 22, 53, 8080])
            dst_port = f"{dst_port_val:05d}"
            
            protocol = random.choice(["06", "17"])  # TCP or UDP
            
            line = f"{src_ip}{dst_ip}{src_port}{dst_port}{protocol}\n"
            f.write(line)
    
    print(f"✓ Created {filename}")

def generate_sample_legitimate_spike_file(filename: str = "sample_spike.txt", num_flows: int = 1000):
    """Generate a sample legitimate traffic spike flow file."""
    print(f"Generating legitimate spike sample: {filename} ({num_flows} flows)")
    
    # Create a power-law distribution of sources (few heavy users, many light users)
    num_sources = max(10, num_flows // 100)
    sources = []
    
    for i in range(num_sources):
        src_ip = f"{random.randint(1, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(1, 254):03d}"
        # Weight decreases for later sources (power law)
        weight = int((num_sources - i) ** 1.5)
        sources.extend([src_ip] * weight)
    
    with open(filename, 'w') as f:
        for i in range(num_flows):
            src_ip = random.choice(sources)
            # Many different destinations
            dst_ip = f"{random.randint(1, 255):03d}{random.randint(0, 255):03d}{random.randint(0, 255):03d}{random.randint(1, 254):03d}"
            src_port = f"{random.randint(1024, 65535):05d}"
            dst_port = random.choice(["00080", "00443"])  # HTTP/HTTPS
            protocol = "06"  # TCP
            
            line = f"{src_ip}{dst_ip}{src_port}{dst_port}{protocol}\n"
            f.write(line)
    
    print(f"✓ Created {filename}")

def generate_all_samples():
    """Generate all sample files."""
    print("=" * 80)
    print("GENERATING SAMPLE FLOW FILES")
    print("=" * 80)
    print()
    
    random.seed(42)  # For reproducibility
    
    generate_sample_ddos_file("sample_ddos.txt", 5000)
    generate_sample_portscan_file("sample_portscan.txt", 2000)
    generate_sample_normal_file("sample_normal.txt", 1000)
    generate_sample_legitimate_spike_file("sample_spike.txt", 3000)
    
    print()
    print("=" * 80)
    print("SAMPLE FILES CREATED")
    print("=" * 80)
    print()
    print("To analyze these files, run:")
    print("  python Entropy_Fixed.py sample_ddos.txt")
    print("  python Entropy_Fixed.py sample_portscan.txt")
    print("  python Entropy_Fixed.py sample_normal.txt")
    print("  python Entropy_Fixed.py sample_spike.txt")
    print()
    print("Or with custom names:")
    print('  python Entropy_Fixed.py sample_ddos.txt "DDoS Attack"')

if __name__ == "__main__":
    generate_all_samples()