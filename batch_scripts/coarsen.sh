#!/bin/bash

### JOB NAME
#PBS -N coarsen

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=4:mem=32GB

### RUNTIME
#PBS -l walltime=04:00:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

### Load modules
module load conda
conda activate s2s

## Run
python -u ~/s2s/data_processing/coarsen.py
