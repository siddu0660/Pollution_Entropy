# runner.py
import ctypes
import numpy as np
import argparse
import os


print("Starting ... ")
# Load the shared library
libhelper = ctypes.CDLL('./parallelLib.so')

# Define the argument types for the C function
libhelper.parallel_runner.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int

]

# get command line args
parser = argparse.ArgumentParser()
parser.add_argument("--itemCapacity",metavar="Expected number of Items",help="Used to set the sixe of IBLT",required= True)
parser.add_argument("--attVar",metavar="Expected number of Items",help="1. QOA 2. CIA 3. Random",required= True)
parser.add_argument("--path",metavar="Path to folder containing files",help="Give absolute path",required=True)

args = parser.parse_args()
items = int(args.itemCapacity)
attVar = int(args.attVar)
folder = args.path

# Getting the values ready
input_dir = folder.replace('.','')
if not os.path.exists(input_dir):
    print('Invalid pcap directory!! Give absolute Path to the folder')
    exit()

#mal_temp = ['0','0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','2','3','4','5','6','7','8','9','10']
mal_temp = ['0','1','2','3','4','5']
malLen = len(mal_temp)

files = os.listdir(input_dir)
for file in files:
    if not file.endswith(".txt"):
        files.remove(file)
numFiles = len(files)
input_dir += '/'

# Formatting to be done before passing to CDLL
mal = (ctypes.c_char_p * len(mal_temp))(*[s.encode('utf-8') for s in mal_temp])
folderPath = ctypes.c_char_p(input_dir.encode('utf-8'))
fileList = (ctypes.c_char_p * len(files))(*[s.encode('utf-8') for s in files])

print("Encoding successfull ... ")
# Set the environment variable to bind threads to processors
import os
os.environ["OMP_PROC_BIND"] = "TRUE"

# Call the C function
try:
    print("C function initialising ... ")
    libhelper.parallel_runner(items, mal, malLen, attVar, folderPath, fileList, numFiles) 
except Exception as e:
    print(e)
# parallel_runner(int itemCapacity, char **mal,int malLen, int attVar, char *path, char **files, int numFiles) 

### COMMAND TO RUN ####
# sudo python3 runner.py --itemCapacity 23988 --attVar 1 --path /home/netx2/SecinfraHarish/attackIBLTFlowRadarMultipleEpoch/txtDataset 
