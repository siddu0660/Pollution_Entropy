import os
import argparse
import csv
import multiprocessing
from functools import partial
import time
import random

cpu_dict = {
    "0": "4",
    "10": "5",
    "0.01":"2",
    "0.02":"3",
    "0.03":"4",
    "0.04":"5",
    "0.05":"6",
    "0.06":"7",
    "0.07":"8",
    "0.08":"9",
    "0.09":"10",
    "0.1":"11",
    "0.2": "12",
    "0.3": "13",
    "0.4": "14",
    "0.5": "15",
    "0.6": "16",
    "0.7": "17",
    "0.8": "18",
    "0.9": "19",
    "1": "20",
    "2": "21",
    "3": "22",
    "4": "23",
    "5": "24",
    "6": "25",
    "7": "26",
    "8": "27",
    "9": "28",
    
}

def runner(items, input_file, mal, var):
    global cpu_dict
    cpu = cpu_dict[mal]
    cpu = str((int(cpu)))
    cmd = "sudo taskset -c "+str(cpu)+" python3 attackIBLTFlowRadarMultipleEpoch.py --items "+str(items)+" --mal "+str(mal)+" --var "+str(var)+" --dataset "+str(input_file)
    print('\n\n Executing : '+cmd)
    os.system(cmd)
    

def for_loop(mal,input_dir, items, var):
    print("Running for --mal "+str(mal))

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            runner(items, file_path, mal, var)


if __name__ =="__main__":
    # reading command line arguments
    parser = argparse.ArgumentParser(description='engine reader')
    parser.add_argument('--items', metavar='<Expected items to set the size of IBLT>', help='number of items to set IBLT size', required=True)
    parser.add_argument('--folder', metavar='<Folder containing generated text files>', help='Provide Absolute path to the folder', required=True)
    parser.add_argument('--var',metavar='<Variant of attack', help='1. QOA, 2. CIA, 3. Random', required=True)
    args = parser.parse_args()

    # assigning command line arguments
    items = int(args.items)
    folder = args.folder
    var = int(args.var)

    #checking validity of given folder 
    input_dir = folder.replace('.','')
    if not os.path.exists(input_dir):
        print('Invalid pcap directory!! Give absolute Path to the folder')
        exit()

    # assigning mal values
    # mal_values = ['0','0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','2','3','4','5','6','7','8','9','10']
    mal_values = ['0','10']

    with multiprocessing.Pool() as pool:
        pool.map(partial(for_loop,input_dir=input_dir, items = items,  var = var),mal_values)