import sys
from scapy.all import PcapReader, IP, TCP, UDP

def format_ip(ip_string):
    """
    Convert '192.168.1.5' -> '192168001005' (12 chars)
    """
    try:
        octets = ip_string.split('.')
        # Format each octet to 3 digits (e.g., '1' -> '001') and join
        return "".join(f"{int(octet):03d}" for octet in octets)
    except ValueError:
        return "000000000000" # Fallback for malformed IPs

def process_pcap(input_file, output_file):
    print(f"Reading {input_file}...")
    
    count = 0
    
    with open(output_file, 'w') as out_f:
        # PcapReader reads packets one-by-one (streaming) to save RAM
        # It handles .gz files automatically
        with PcapReader(input_file) as pcap:
            for packet in pcap:
                
                # We only care about IP packets
                if IP in packet:
                    ip_layer = packet[IP]
                    
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto = ip_layer.proto
                    
                    # Default ports to 0
                    src_port = 0
                    dst_port = 0
                    
                    # Extract ports if TCP or UDP
                    if TCP in packet:
                        src_port = packet[TCP].sport
                        dst_port = packet[TCP].dport
                    elif UDP in packet:
                        src_port = packet[UDP].sport
                        dst_port = packet[UDP].dport
                    
                    # Format fields to specific widths
                    # IP: 12 chars (e.g., 192168001001)
                    fmt_src_ip = format_ip(src_ip)
                    fmt_dst_ip = format_ip(dst_ip)
                    
                    # Port: 5 chars (e.g., 00080)
                    fmt_src_port = f"{src_port:05d}"
                    fmt_dst_port = f"{dst_port:05d}"
                    
                    # Protocol: 2 chars (e.g., 06)
                    fmt_proto = f"{proto:02d}"
                    
                    # Construct line: 12 + 12 + 5 + 5 + 2 = 36 chars
                    line = f"{fmt_src_ip}{fmt_dst_ip}{fmt_src_port}{fmt_dst_port}{fmt_proto}\n"
                    
                    out_f.write(line)
                    count += 1

    print(f"Conversion complete. {count} lines written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pcapToFlows.py <input.pcap.gz> <output.txt>")
    else:
        process_pcap(sys.argv[1], sys.argv[2])