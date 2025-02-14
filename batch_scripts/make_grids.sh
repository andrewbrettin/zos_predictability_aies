#!/bin/bash

### JOB NAME
#PBS -N grids

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=8:mem=64GB

### RUNTIME
#PBS -l walltime=00:10:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

module load conda
conda activate npl

python -u ~/s2s/data_processing/make_grids.py

## Move output and error files 
mv $PBS_JOBNAME.e* errors
mv $PBS_JOBNAME.o* output
