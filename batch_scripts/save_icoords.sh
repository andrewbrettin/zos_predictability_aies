#!/bin/bash
### JOB NAME
#PBS -N icoords

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=16GB

### RUNTIME
#PBS -l walltime=00:10:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

### OUTOUT
#PBS -o output/
#PBS -e errors/

### Variables
TAU=60

### Echo configurations
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}


### Load modules
module load conda
conda activate s2s

### Run
printf "\nMAKE ICOORDS\n"
python -u ~/s2s/data_processing/save_icoords.py ${TAU}