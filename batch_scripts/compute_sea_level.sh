#!/bin/bash

### JOB NAME
#PBS -N sea_level

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=8GB

### RUNTIME
#PBS -l walltime=03:00:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

module load conda
conda activate s2s

python -u ~/s2s/data_processing/compute_sea_level_variables.py

## Move output and error files 
mv $PBS_JOBNAME.e* errors
mv $PBS_JOBNAME.o* output
