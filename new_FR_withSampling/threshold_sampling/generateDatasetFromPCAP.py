from scapy.all import *
import argparse
import sys
import random


def ipAddressPadder(ip):
    padded_ip_address = ''.join(octet.zfill(3) for octet in ip.split('.'))
    return padded_ip_address

def store_items_to_file(items, filename):
    with open(filename, 'w') as file:
        for item in items:
            file.write(str(item) + '\n')



##### MAIN FUNCTION FLOW########
parser = argparse.ArgumentParser(description='Input from the user')
parser.add_argument('--pcap', metavar='<Pcap file path to be used as input>', help='Pcap file to be inserted into the IBLT', required=True)
args = parser.parse_args()

pcap_file = args.pcap

flows = set()

# Reading PCAP file
packets = PcapReader(pcap_file)

# Output file name based on the PCAP file name 
op_file = pcap_file.split('/')[-1].split('.')[0]
op_file = op_file + '.txt'
print(f'OUTPUT CREATED IN FILE : {op_file}')

# Processing each packet
unique_flow_ids = set()


srcIp = ''
dstIP = ''
srcPort = ''
dstPort = ''
proto = ''



for pkt in packets:
    # skipping IPv6 packets
    if IPv6 in pkt:
        continue

    #Finding the flow ID for IPv4 packets
    if IP in pkt and TCP in pkt:
        srcIp = str(pkt[IP].src)
        dstIP= str(pkt[IP].dst)
        srcPort= str(pkt[TCP].sport)
        dstPort= str(pkt[TCP].dport)
        proto= str(pkt[IP].proto )
	
    elif IP in pkt and UDP in pkt:
        srcIp = str(pkt[IP].src)
        dstIP=str(pkt[IP].dst)
        srcPort=str(pkt[UDP].sport)
        dstPort=str(pkt[UDP].dport)
        proto=str(pkt[IP].proto)
        
    else:
        srcIp = str(pkt[IP].src)
        dstIP=str(pkt[IP].dst)
        srcPort=str(0)
        dstPort=str(0)
        proto=str(pkt[IP].proto)



	##padd the IP addresses
    srcIp = ipAddressPadder(srcIp)
    dstIP = ipAddressPadder(dstIP)
    srcPort = srcPort.zfill(5)
    dstPort = dstPort.zfill(5)
    proto = proto.zfill(2)
    
    flowID = srcIp + dstIP + srcPort + dstPort + proto
  
    unique_flow_ids.add(flowID)
    print(len(unique_flow_ids))

store_items_to_file(unique_flow_ids, op_file)


