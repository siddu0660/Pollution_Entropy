// helper.c
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void logToCSV(char *filename, char *mal, int status) {
    FILE *fp;
    fp = fopen("logfile.csv", "a"); // Open the CSV file in append mode

    if (fp == NULL) {
        perror("Error opening file");
        exit(-1);
    }

    // Write the status and message to the CSV file
    fprintf(fp, "%s,%s,%d\n",filename,mal, status);

    fclose(fp);
}

void parallel_runner(int itemCapacity, char **mal,int malLen, int attVar, char *path, char **files, int numFiles) {
    printf("Entering Parallel Region ....\n");
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < malLen; i++){
        for (int j = 0; j < numFiles; j++){
            char command[1000];
            char dataset[1000];
            strcpy(dataset, path);
            strcat(dataset, files[j]);
            snprintf(command, sizeof(command), "sudo python3 attackIBLTFlowRadarMultipleEpoch.py --items %d --mal %s --var %d --dataset %s", itemCapacity, mal[i], attVar, dataset);
            printf("[+] Running : %s with mal %s on core %d thread %d\n",files[j],mal[i],sched_getcpu(),omp_get_thread_num());
            int status = system(command);
            char message[100];
            if (status == -1) {
                strcpy(message,"Error executing command");
            } 
            else {
                if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                    strcpy(message,"Command executed successfully");
                } 
                else {
                    strcpy(message,"Command failed");
                }
            }
            printf("[+] COMPLETED : %s with mal %s with exit code %d\n",files[j],mal[i],status);
            logToCSV(files[j], mal[i], status);
        }
    }
}

