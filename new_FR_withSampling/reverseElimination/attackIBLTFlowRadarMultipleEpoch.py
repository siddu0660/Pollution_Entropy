from scapy.all import *
import argparse
import sys
import mmh3 #Murmur3 Hash function.
import csv
import os
import time


############Global Variables

###Bloom filter Variable
bfarray = [0]


############IBLT Variables
keySum = ["0".zfill(36)] ##IBLT Variable
count = [0] ##IBLT Variable
valueSum = [0] ##IBLT Variable

############functional variables
extractedValues = [] ##contains the extraced / decoded elements from the IBLT
inputList = [] ##contains the list of input elements read from the dataset
maliciousItems = [] ##contains the malicious items

############Feature / statistics variables

all_decodable_count = 0 #done
all_undecodable_count = 0 #done
all_decodable_percent = 0.0 #done
all_undecodable_percent = 0.0 #done

benign_decodable_count = 0 #done
benign_undecodable_count = 0 #done
benign_decodable_percent = 0.0 #done
benign_undecodable_percent = 0.0 #done

purecell_count_before_decode = 0 #done
purecell_count_after_decode = 0 #done
purecell_all_collision = 0  #done
purecell_ben_collision = 0 #done
purecell_mal_collision = 0 #done
completely_colliding_flows = 0 #done
completely_colliding_mal_flows = 0 #done
completely_colliding_ben_flows = 0 #done
non_colliding_flows = 0 #done
non_colliding_mal_flows = 0 #done
non_colliding_ben_flows = 0 #done

collisions_by_benignflows = 0 #done
collisions_by_maliciousflows = 0 #done
collisions_by_allflows = 0 #done

cells_occupied_epochend = 0 #done
cells_occupied_decodeend = 0 #done
cells_occupied_epochend_percent = 0.0 #done
cells_occupied_decodeend_percent = 0.0 #done

catergory_of_flows = '' #done

no_of_benign_flows_classified_as_old = 0 #done
no_of_benign_flows_classified_as_new = 0 #done
no_of_malicious_flows_classified_as_old = 0 #done
no_of_malicious_flows_classified_as_new = 0 #done
total_flows_classified_as_old = 0 #done
total_flows_classified_as_new = 0 #done


###### Pollution mitigation variables ##############

sampled_flows = [] # sampling : implemented --> Mitigation : implemented
pre_mit_decodable = 0 # done
pre_mit_undecodable = 0 # done
pre_mit_decodable_percent = 0.0 # done
pre_mit_undecodable_percent = 0.0 # done
candidateFlowCount = 0 # done
flowsUsedForMitigation = 0 # done

############UTILITY FUNCTIONS############
####################################

###########Function to XOR Two strings
def xor_strings(str1, str2):
    result = hex(int(str1, 16) ^ int(str2, 16))[2:].zfill(36)
    return result


############Check purecell count############
def checkPurecell(sizeIBLT):
    global count
    purecells= count.count(1)
    # print("Purecell count = {}".format(purecells)) 
    return purecells
    # for i in range(0,sizeIBLT):
    #     if count[i] == 1:
    #         purecells=purecells+1
        # print("{} = {}".format(keySum[i],count[i]))

############Check IBLT occupancy statistics############
def checkOccupancy(sizeIBLT):
    global count
    unoccupiedcells = count.count(0)
    occupiedcells = sizeIBLT - unoccupiedcells
    percentofoccupied = float((occupiedcells / sizeIBLT) * 100)

    return occupiedcells, percentofoccupied

############ Generate Random Flows ############
def generateRandomFlow():
    src_IP = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
    src_IP = ''.join(octet.zfill(3) for octet in src_IP.split('.'))
    dst_IP = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
    dst_IP = ''.join(octet.zfill(3) for octet in dst_IP.split('.'))
    src_port = str(random.randint(0,65535)).zfill(5)
    dst_port = str(random.randint(0,65535)).zfill(5)
    proto = random.choice(['06', '17']) # TCP,UDP

    flowID = src_IP + dst_IP + src_port + dst_port + proto
    return flowID

    

     


############CREATE ATTACK ITEMS############
#####################################
def createMalicious(exptVariant,malFlows,inputItems):
    global maliciousItems
    noOfMalFlows = int(inputItems * (malFlows / 100)) #calculating the number of malicious flows

    ###Type I QoA Attack
    if exptVariant == 1:
        # print("Type I: QoA")

        ##proportional to the malicious flow %, gather a subset of input elements to create the image

        ##But for now, we are considering the whole input set as the subset 
        ##get the list of indices
        hashIndicesImage = set()
        size = inputItems
        size = int(size) #/10
        k=0
        for item in inputList:
            ##Calculate hash indices
            k+=1 
            if k == size:
                break

            for i in range(0,hashIBLT):
                hashVal = mmh3.hash(str(item),i)
                hashIndicesImage.add(int(abs(hashVal)) % sizeIBLT) ##input the hash indices into the set 
        
        # print(hashIndicesImage)
        unique_items = set() ##generate unique random flows
        counter=0
        while len(unique_items) < noOfMalFlows:
            generatedItem = generateRandomFlow()
            # print("{} {}".format(counter,len(unique_items)),end='\r')
            counter=counter+1
            #calculate indices of generated Item
            flag = True
            for i in range(0,hashIBLT): #calculate the indices for the purecell's value and extract them as well
                hashVal = mmh3.hash(str(generatedItem),i)
                hashIndex = int(abs(hashVal)) % sizeIBLT
                if hashIndex not in hashIndicesImage:
                    flag = False
                    break

            ##if break did not happen, then the indice is good to go and store the elemene in unique_items    
            if flag:
                unique_items.add(generatedItem)

        maliciousItems = list(unique_items) 
        return noOfMalFlows


    elif exptVariant == 2:
        # print("CIA")

        unique_items = set() ##generate unique random flows
        generatedHashIndices = list()
        while len(unique_items) < noOfMalFlows:
            generatedItem = generateRandomFlow()
            #calculate indices of generated Item
            flag = True
            for i in range(0,hashIBLT): #calculate the indices for the purecell's value and extract them as well
                hashVal = mmh3.hash(str(generatedItem),i)
                hashIndex = int(abs(hashVal)) % sizeIBLT
                if hashIndex in generatedHashIndices:
                    flag = False
                    break

            ##if break did not happen, then the indice is good to go and store the elemene in unique_items    
            if flag:
                unique_items.add(generatedItem)

        maliciousItems = list(unique_items) 
        return noOfMalFlows

    elif exptVariant ==3:
        # print("Random")
        unique_items = set() ##generate unique random flows
        while len(unique_items) < noOfMalFlows:
            unique_items.add(generateRandomFlow())

        maliciousItems = list(unique_items) 

        # with open('Malicious.txt','w') as file:
        #      writer =csv.writer(file)
        #      for i in maliciousItems:
        #           writer.writerow([i])
        
        return noOfMalFlows
    ######## Commenting out the logic to keep track of malicious flows in a file ########    
        
        """
        original_stdout = sys.stdout
        output_file_path = "maliciousflows.txt"

        try:
            # Open the file in write mode and redirect the standard output stream to it
            with open(output_file_path, 'w') as file:
                sys.stdout = file
                unique_items = set() ##generate unique random flows
                while len(unique_items) < noOfMalFlows:
                    unique_items.add(random.randint(10000000000, 99999999999))

                maliciousItems = list(unique_items) 
                for item in unique_items: #write the generated malicious flows to the file
                    file.write(str(item) + '\n')
        finally:
            sys.stdout = original_stdout   
        """
    
############IBLT FUNCTONS############
#####################################

############Function to perform SingleDecode#############
def singleDecode(sizeIBLT, hashIBLT, inputItems, noOfMalFlows):
    # print("inside singleDecode")
    global extractedValues
    global all_decodable_count, all_undecodable_count, all_decodable_percent,all_undecodable_percent, purecell_count_after_decode
    global benign_decodable_count, benign_undecodable_count, benign_decodable_percent, benign_undecodable_percent
    global sampled_flows, pre_mit_decodable, pre_mit_undecodable, pre_mit_decodable_percent, pre_mit_undecodable_percent
    global candidateFlowCount, flowsUsedForMitigation



    encountered = True ##Variable to determine if any purecells are remaining in the IBLT
    counter=0
    nonMitigatedEnd = True
    
    while encountered or len(sampled_flows) > 0:
        
        if not encountered: # this ensures additional decoding without mitigation concept
            if nonMitigatedEnd: # finding actual number of flows required
                candidateFlowCount = len(sampled_flows)
                nonMitigatedEnd = False
            # pre-mitigation telemetry
            totalflows_including_malicious = inputItems+ noOfMalFlows
            pre_mit_decodable = len(extractedValues)
            pre_mit_undecodable = totalflows_including_malicious - pre_mit_decodable
            pre_mit_decodable_percent = (pre_mit_decodable / totalflows_including_malicious) * 100
            pre_mit_undecodable_percent = (pre_mit_undecodable / totalflows_including_malicious) * 100

            extractedValue = sampled_flows[-1] # choose a sampled value to be extracted
            flowsUsedForMitigation += 1
            sampled_flows.remove(extractedValue) # remove chosen value from the sampled flows
            extractedValues.append(str(extractedValue)) ##store the item in memory

            for j in range(0,hashIBLT): #calculate the indices for the value and extract them as well
                    hashVal = mmh3.hash(str(extractedValue),j)
                    hashIndex = int(abs(hashVal)) % sizeIBLT #get the hash indice
                    ##Delete the elements from the indexes calculated 
                    keySum[hashIndex] = xor_strings(keySum[hashIndex],extractedValue) 
                    count[hashIndex] = count[hashIndex] - 1

        encountered = False

        for i in range(0,sizeIBLT):
            
            if count[i] == 1: #if purecell found, extract value and calculate other indices and extract them as well
                encountered = True #if a pure cells is found, then continue the loop
                extractedValue = keySum[i] #note the value
                if extractedValue in sampled_flows: # remove already extracted value from the sampled flows
                    sampled_flows.remove(extractedValue) 
                extractedValues.append(str(extractedValue)) ##store the item in memory

                for j in range(0,hashIBLT): #calculate the indices for the purecell's value and extract them as well
                    hashVal = mmh3.hash(str(extractedValue),j)
                    hashIndex = int(abs(hashVal)) % sizeIBLT #get the hash indice
                    ##Delete the elements from the indexes calculated 
                    keySum[hashIndex] = xor_strings(keySum[hashIndex],extractedValue) 
                    count[hashIndex] = count[hashIndex] - 1
    
    ## getting the statistics of decodable and undecodable with respect to all flows seen
    # post-mitigation telemetry
    totalflows_including_malicious = inputItems+ noOfMalFlows
    all_decodable_count = len(extractedValues)
    all_undecodable_count = totalflows_including_malicious - all_decodable_count
    all_decodable_percent = (all_decodable_count / totalflows_including_malicious) * 100
    all_undecodable_percent = (all_undecodable_count / totalflows_including_malicious) * 100

    
    # print("##### Verification of SingleDecode")
    # print("Number of Input Elements = {}".format(len(inputList)))
    # print("Number of Extracted Elements = {}".format(len(extractedValues)))
    counter = 0
    for i in range(0,len(extractedValues)):
        for j in range(0,len(inputList)):
            if extractedValues[i] == inputList[j]:
                counter+=1
                break
    # print("Number of Matched Elements = {}".format(counter))
    purecell_count_after_decode = checkPurecell(sizeIBLT)

    ##### getting the statistics of decodable and undecodable with respect to only benign flows
    totalflows_only_benign = inputItems
    benign_decodable_count = counter
    benign_undecodable_count = totalflows_only_benign - benign_decodable_count
    benign_decodable_percent = (benign_decodable_count / totalflows_only_benign) * 100
    benign_undecodable_percent = (benign_undecodable_count / totalflows_only_benign) * 100


############Function to insert an element into the IBLT#############
def insertIBLT(val,hashIBLT,Mal):
    # print("inside insertIBLT")
    global keySum,bfarray
    global count
    global purecell_all_collision, purecell_ben_collision, purecell_mal_collision
    global collisions_by_allflows, collisions_by_benignflows, collisions_by_maliciousflows
    global completely_colliding_flows, non_colliding_flows
    global completely_colliding_mal_flows, completely_colliding_ben_flows
    global non_colliding_mal_flows, non_colliding_ben_flows
    global no_of_benign_flows_classified_as_old,no_of_benign_flows_classified_as_new,no_of_malicious_flows_classified_as_old
    global no_of_malicious_flows_classified_as_new,total_flows_classified_as_old,total_flows_classified_as_new
    global sampled_flows


    ######insert into the bloom filter#######

    ####get the BF hash indices######
    hashIndiceBF = [0] * hashBF    
    ##Calculate hash indices
    for i in range(0,hashBF):
        hashVal = mmh3.hash(str(val),i)
        hashIndiceBF[i] = int(abs(hashVal)) % sizeBF #get the hash indice 

    newFlow = False
    for indice in hashIndiceBF:
        if bfarray[indice] == 0:
           newFlow = True
           bfarray[indice] = 1



    if not newFlow:
        total_flows_classified_as_old+=1
        if Mal:
            no_of_malicious_flows_classified_as_old+=1
        else:
            no_of_benign_flows_classified_as_old+=1
        return
    else:
        total_flows_classified_as_new+=1
        if Mal:
            no_of_malicious_flows_classified_as_new+=1
        else:
            no_of_benign_flows_classified_as_new+=1

    hashIndice = [0] * hashIBLT
    ##Calculate hash indices
    for i in range(0,hashIBLT):
        hashVal = mmh3.hash(str(val),i)
        hashIndice[i] = int(abs(hashVal)) % sizeIBLT #get the hash indice 

    ##Enable to print the values    
    # print("{} = {} = {}".format(val,hashVal,hashIndice))

    ##for each of the hashfunction, insert at the appropriate index (this is redundant but okay. remove later)
    collisionCounterPerItem=0
    for i in range(0,hashIBLT):
          
        if count[hashIndice[i]] != 0: ##if the cell is occupied

            collisionCounterPerItem+=1  ## increment collision count per item to determine completely colliding and non-colliding flows
            collisions_by_allflows+=1 ## statistics for total collisions on purecells
            if Mal:
               collisions_by_maliciousflows+=1 ## statistics for number of malicious collisions on purecells
            else:
               collisions_by_benignflows+=1 ## statistics for number of benign collisions on purecells


        if count[hashIndice[i]] == 1: ##if the cell is a pure cell
            purecell_all_collision+=1 ## statistics for total collisions on purecells
            if Mal:
               purecell_mal_collision+=1 ## statistics for number of malicious collisions on purecells
            else:
               purecell_ben_collision+=1 ## statistics for number of benign collisions on purecells
        
        # keySum[hashIndice[i]] = str(hex(int(keySum[hashIndice[i]],16) ^ int(str(val), 16)))[2:].zfill(26) 
        
        keySum[hashIndice[i]] = xor_strings(keySum[hashIndice[i]],val) 
        # print(keySum[hashIndice[i]])
        count[hashIndice[i]] = count[hashIndice[i]] + 1
    
    if collisionCounterPerItem == hashIBLT:
        sampled_flows.append(val)
        completely_colliding_flows+=1
        if Mal:
            completely_colliding_mal_flows+=1
        else:
            completely_colliding_ben_flows+=1
    if collisionCounterPerItem == 0:
        non_colliding_flows+=1
        if Mal:
            non_colliding_mal_flows+=1
        else:
            non_colliding_ben_flows+=1

############Function to create the IBLT#############
def createIBLT(sizeIBLT):
    # print("inside Create IBLT")
    
    #######IBLT Created
    global keySum
    global count
    global valueSum
    keySum = ["0".zfill(42)] * sizeIBLT
    count = [0] * sizeIBLT
    valueSum = [0] * sizeIBLT

############Function to create the BF#############
def createBF(sizeBF):
    global bfarray
    bfarray = [0] * sizeBF


####################################################################
####################################################################
############MAIN FLOW OF OF THE PROGRAM#############################
####################################################################
####################################################################

parser = argparse.ArgumentParser(description='Input from the user')
parser.add_argument('--items', metavar='<no of items expected>', help='Items to be inserted into the IBLT', required=True)
parser.add_argument('--mal', metavar='<percent of malicious to be crafted>', help='percent of malicious to be crafted', required=True)
parser.add_argument('--var', metavar='<variant of the adversary>', help='varaiant of attack; 1 = QoA, 2 = CIA, 3 = RND', required=True)
parser.add_argument('--dataset', metavar='<input file name>', help='input file to parse', required=True)
args = parser.parse_args()


############Input Arguments
inputItems = int(args.items) #Expected number of input items
malFlows = float(args.mal) #Enter percentage of malicious items to be crafted
exptVariant = int(args.var) #varaiant of adversary; 1 = QoA, 2 = CIA, 3 = RND
dataset = args.dataset

epoch_name = dataset.split('/')[-1].split('.')[0]
malStr = args.mal.replace('.','')

############Additional Arguments
hashIBLT = 4 #the number of IBLT hash functions
sizeIBLT = int(inputItems * 1.295)   #1.295

##########Bloomfilter properties
error_rate = 0.01
hashBF = int(math.ceil(math.log(1 / error_rate, 2)))
bits_per_slice = int(math.ceil( (inputItems * abs(math.log(error_rate))) / (hashBF * (math.log(2) ** 2))))
sizeBF = hashBF * bits_per_slice
# dataSet = args.pcap #the name of the input file
# fpr = 0.01 #the FPR of the BF

# print("######Input Parameters of the Experiment######")
# print('Input Items: {}'.format(inputItems))
# print('Malicious Flows: {}'.format(malFlows))
# print('Experiment Variant: {}'.format(exptVariant))
# print('Input Dataset: {}'.format(dataset))
# print("######Additional Parameters of the Experiment######")
# print('Hash Functions CT: {}'.format(hashIBLT))
# print('IBLT Size: {}'.format(sizeIBLT))
# print('Hash Functions BF: {}'.format(hashBF))
# print('BF Size: {}'.format(sizeBF))


"""
############Print the arguments into a file

original_stdout = sys.stdout
output_file_path = "parameters.txt"

try:
    # Open the file in write mode and redirect the standard output stream to it
    with open(output_file_path, 'w') as file:
        sys.stdout = file
        print("######Input Parameters of the Experiment######")
        print('Input Items: {}'.format(inputItems))
        print('Malicious Flows: {}'.format(malFlows))
        print('Experiment Variant: {}'.format(exptVariant))
        print('Input Dataset: {}'.format(dataset))
        print("######Additional Parameters of the Experiment######")
        print('Hash Functions: {}'.format(hashIBLT))
        print('IBLT Size: {}'.format(sizeIBLT))
finally:
    sys.stdout = original_stdout
"""
############Create the IBLT
createBF(sizeBF)
createIBLT(sizeIBLT)

############Insert Items into the IBLT
##reading elements from the dataset
with open(dataset, 'r') as file:
    #First we read the elements and then we insert
    for line in file:
        val = line.strip() #Print each line after stripping newline characters. The type is string
        inputList.append(val) #add the dataset to an in memory list

if inputItems < len(inputList):
    catergory_of_flows = 'SPIKE'
elif len(inputList) < int(0.9 * inputItems):
    catergory_of_flows = 'DROP'
else:
    catergory_of_flows = 'NORMAL'

capacityParameter = inputItems
inputItems = len(inputList)


############Attack the IBLT as per adversary variant
noOfMalFlows = createMalicious(exptVariant,malFlows,inputItems)



       
##Now Insert the element into the IBLT
insertionsNotOver = True
probability = 0.1
malIterator=0
benIterator=0


while insertionsNotOver:
    if random.random() < probability and  malIterator < noOfMalFlows:  #noOfMalFlows > 0 and
        # print("Insert MalFlows")
        val = maliciousItems[malIterator]
        insertIBLT(val,hashIBLT,True)
        malIterator+=1
    elif benIterator < inputItems:
        # print("Insert Benign Flows")
        # print(benIterator)
        val = inputList[benIterator]
        insertIBLT(val,hashIBLT,False)
        benIterator+=1

    ## Check if insertions are over. If so, then break out
    totalSize = noOfMalFlows + inputItems
    itemsInserted = malIterator + benIterator
    if itemsInserted == totalSize: ##If all the elements have been inserted, then exit the loop
        insertionsNotOver = False


# print('Number of Malicious flows: {}'.format(noOfMalFlows))

          
############Check the IBLT (For future statistics, write the code here)
purecell_count_before_decode = checkPurecell(sizeIBLT)

###########Get the occupancy statistics of IBLT at epoch end##########
cells_occupied_epochend, cells_occupied_epochend_percent = checkOccupancy(sizeIBLT)


# print('Number of Purecell collision by all flows: {}'.format(purecell_all_collision))
# print('Number of Purecell collision by malicious flows: {}'.format(purecell_mal_collision))
# print('Number of Purecell collision by benign flows: {}'.format(purecell_ben_collision))

# print('Number of collisions by all flows: {}'.format(collisions_by_allflows))
# print('Number of collisions by malicious flows: {}'.format(collisions_by_maliciousflows))
# print('Number of collisions by benign flows: {}'.format(collisions_by_benignflows))

# print('Total Number of completely colliding flows: {}'.format(completely_colliding_flows))
# print('Number of completely colliding malicious flows: {}'.format(completely_colliding_mal_flows))
# print('Number of completely colliding benign flows: {}'.format(completely_colliding_ben_flows))

# print('Total Number of non colliding flows: {}'.format(non_colliding_flows))
# print('Number of non colliding malicious flows: {}'.format(non_colliding_mal_flows))
# print('Number of non colliding benign flows: {}'.format(non_colliding_ben_flows))

numSampled = len(sampled_flows)

start = time.time()
############Perfrom SingleDecode process
singleDecode(sizeIBLT, hashIBLT, inputItems, noOfMalFlows)
end = time.time()


###########Get the occupancy statistics of IBLT after decode##########
cells_occupied_decodeend, cells_occupied_decodeend_percent = checkOccupancy(sizeIBLT)


# print('Number of cells occupied epochend: {}'.format(cells_occupied_epochend))
# print('percentage of cells occupied epochend: {}'.format(cells_occupied_epochend_percent))
# print('Number of cells occupied decodeend: {}'.format(cells_occupied_decodeend))
# print('percentage of cells occupied decodeend: {}'.format(cells_occupied_decodeend_percent))


# print('Considering all: decodable items {}'.format(all_decodable_count))
# print('Considering all: undecodable items {}'.format(all_undecodable_count))
# print('Considering all: decodable percent {}'.format(all_decodable_percent))
# print('Considering all: undecodable percent {}'.format(all_undecodable_percent))

# print('Considering benign: decodable items {}'.format(benign_decodable_count))
# print('Considering benign: undecodable items {}'.format(benign_undecodable_count))
# print('Considering benign: decodable percent {}'.format(benign_decodable_percent))
# print('Considering benign: undecodable percent {}'.format(benign_undecodable_percent))

# print('Purecells before decode: {}'.format(purecell_count_before_decode))
# print('Purecells after decode: {}'.format(purecell_count_after_decode))


# print('No of benign flows classified as old {}'.format(no_of_benign_flows_classified_as_old))
# print('No of benign flows classified as new {}'.format(no_of_benign_flows_classified_as_new))
# print('No of malicious flows classified as old {}'.format(no_of_malicious_flows_classified_as_old))
# print('No of malicious flows classified as new {}'.format(no_of_malicious_flows_classified_as_new))
# print('Total flows classified as old {}'.format(total_flows_classified_as_old))
# print('Total flows classified as new {}'.format(total_flows_classified_as_new))





# print("Total Time taken {}".format(end-start))

#####################################################
###############ADD Results to a CSV#################

folder = ''
filename = ''
if exptVariant == 1:
    folder = './results/TypeI'
    filename = '/'+malStr+'.csv'
elif exptVariant == 2:
    folder = './results/TypeII'
    filename = '/'+malStr+'.csv'
elif exptVariant == 3:
    folder = './results/TypeIII'
    filename = '/'+malStr+'.csv'

os.makedirs(folder, exist_ok=True)

output_file = folder + filename

polluted = 'No'

if all_undecodable_percent > 40.0:
    polluted = 'YES'



if not os.path.isfile(output_file):
    try:
        with open(output_file,'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerows([
                ['Capacity of Flows : ',str(capacityParameter)],
                ['Size of IBLT : ', str(sizeIBLT)],
                ['Number of BF Hash Functions : ',str(hashBF), 'Size of BF : ', str(sizeBF)],
                [
                str('Epoch name'),
                str('noOfMalFlows'),
                str('totalBenignFlows'),
                str('all_decodable_count'),
                str('all_decodable_percent'),
                str('all_undecodable_count'),
                str('all_undecodable_percent'),
                str('benign_decodable_count'),
                str('benign_decodable_percent'),
                str('benign_undecodable_count'),
                str('benign_undecodable_percent'),
                str('purecell_ben_collision'),
                str('purecell_mal_collision'),
                str('purecell_all_collision'),
                str('completely_colliding_flows'),
                str('completely_colliding_mal_flows'),
                str('completely_colliding_ben_flows'),
                str('non_colliding_flows'),
                str('non_colliding_mal_flows'),
                str('non_colliding_ben_flows'),
                str('collisions_by_benignflows'),
                str('collisions_by_maliciousflows'),
                str('collisions_by_allflows'),
                str('cells_occupied_epochend'),
                str('cells_occupied_epochend_percent'),
                str('cells_occupied_decodeend'),
                str('cells_occupied_decodeend_percent'),
                str('purecell_count_before_decode'),
                str('purecell_count_after_decode'),
                str('no_of_benign_flows_classified_as_old'),
                str('no_of_benign_flows_classified_as_new'),
                str('no_of_malicious_flows_classified_as_old'),
                str('no_of_malicious_flows_classified_as_new'),
                str('total_flows_classified_as_old'),
                str('total_flows_classified_as_new'),
                str('catergory_of_flows'),
                str('Pollution ?'),
                str('pre_mit_decodable'),
                str('pre_mit_undecodable'),
                str('pre_mit_decodable_percent'),
                str('pre_mit_undecodable_percent'),
                str('NumberofSampledFlows'),
                str('Sampled Flows after nonMitigated SD'),
                str('Flows used for mitigation')
                ]

            ]) 

    except Exception as e:
        print(e)


try:
    with open(output_file,'a') as result_file:
        writer = csv.writer(result_file)
        writer.writerow([
            str(epoch_name),
            str(noOfMalFlows),
            str(inputItems),
            str(all_decodable_count),
            str(all_decodable_percent),
            str(all_undecodable_count),
            str(all_undecodable_percent),
            str(benign_decodable_count),
            str(benign_decodable_percent),
            str(benign_undecodable_count),
            str(benign_undecodable_percent),
            str(purecell_ben_collision),
            str(purecell_mal_collision),
            str(purecell_all_collision),
            str(completely_colliding_flows),
            str(completely_colliding_mal_flows),
            str(completely_colliding_ben_flows),
            str(non_colliding_flows),
            str(non_colliding_mal_flows),
            str(non_colliding_ben_flows),
            str(collisions_by_benignflows),
            str(collisions_by_maliciousflows),
            str(collisions_by_allflows),
            str(cells_occupied_epochend),
            str(cells_occupied_epochend_percent),
            str(cells_occupied_decodeend),
            str(cells_occupied_decodeend_percent),
            str(purecell_count_before_decode),
            str(purecell_count_after_decode),
            str(no_of_benign_flows_classified_as_old),
            str(no_of_benign_flows_classified_as_new),
            str(no_of_malicious_flows_classified_as_old),
            str(no_of_malicious_flows_classified_as_new),
            str(total_flows_classified_as_old),
            str(total_flows_classified_as_new),
            str(catergory_of_flows),
            str(polluted),
            str(pre_mit_decodable),
            str(pre_mit_undecodable),
            str(pre_mit_decodable_percent),
            str(pre_mit_undecodable_percent),
            str(numSampled),
            str(candidateFlowCount),
            str(flowsUsedForMitigation)

        ]) 

except Exception as e:
    print(e)


print(f"done one run of {malFlows}")