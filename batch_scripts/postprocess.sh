#!/bin/bash
### JOB NAME
#PBS -N postproc

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=16GB

### RUNTIME
#PBS -l walltime=01:00:00

### QUEUE
#PBS -q casper

### JOB ARRAY
##PBS -J 60-180:30

### MESSAGING
#PBS -m ea
#PBS -M <email>

### Variables
#TAU=${PBS_ARRAY_INDEX}
TAU=120
ICOORD=5013

### Echo configurations
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}
echo ${TAU}
echo ${ICOORD}


### Load modules
module load conda
conda activate s2s

### Run
printf "\nPOSTPROCESSING\n"
python -u ~/s2s/data_processing/postprocess.py ${TAU} ${ICOORD}