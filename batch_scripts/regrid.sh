#!/bin/bash

### JOB NAME
#PBS -N regrid

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=8:mem=64GB

### RUNTIME
#PBS -l walltime=02:30:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

module load conda
conda activate npl

python -u ~/s2s/data_processing/regrid.py
