#!/bin/bash
### JOB NAME
#PBS -N tensors

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=2:mem=64GB

### RUNTIME
#PBS -l walltime=03:00:00

### QUEUE
#PBS -q casper

### JOB ARRAY

### MESSAGING
#PBS -m ea
#PBS -M <email>
#PBS -J 60-180:30

### Variables
TAU=${PBS_ARRAY_INDEX}

### Echo configurations
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}


### Load modules
module load conda
conda activate s2s

### Run
printf "\nMAKE TENSORS\n"
python -u ~/s2s/data_processing/make_dp_tensors.py ${TAU}