from scapy.all import *
import argparse
import sys
import mmh3 #Murmur3 Hash function.
import csv
import os

def xor_two_str(a,b):
    xored = []
    for i in range(max(len(a), len(b))):
        xored_value = ord(a[i%len(a)]) ^ ord(b[i%len(b)])
        xored.append(hex(xored_value)[2:])
    return ''.join(xored)

###########Function to XOR Two strings
def xor_strings(str1, str2):
    # Ensure both strings are of equal length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")

    # Perform XOR operation on each corresponding character
    result = ""
    for char1, char2 in zip(str1, str2):
        xor_result = ord(char1) ^ ord(char2)  # XOR operation on ASCII values
        result += chr(xor_result)  # Convert result back to character
    return result


error_rate = 0.01
hashBF = int(math.ceil(math.log(1 / error_rate, 2)))
print(hashBF)

exit()

# keySum = "0".zfill(36)
keySum = "0".zfill(37)
str1= "009179152249030198215187446900069206"
str2= "039114107024017243068078229891877006"

print(keySum)
print(str1)
print(str2)

res1 = hex(int(keySum, 16) ^ int(str1, 16))[2:].zfill(37)


# res1 = xor_two_str(keySum,str1)
print(res1)
# res2 = xor_two_str(res1,str2)
res2 = hex(int(res1, 16) ^ int(str2, 16))[2:].zfill(37)


print(res2)
# res3 = xor_two_str(res2,str2)
res3 = hex(int(res2, 16) ^ int(str1, 16))[2:].zfill(37)

print(res3)

if res3==str2:
    print("true")

error_rate = 0.01
hashBF = int(math.ceil(math.log(1 / error_rate, 2)))
print(hashBF)