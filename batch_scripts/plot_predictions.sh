#!/bin/bash

### JOB NAME
#PBS -N plot_preds

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=8GB

### RUNTIME
#PBS -l walltime=01:00:00

### QUEUE
#PBS -q casper

### MESSAGING
#PBS -m ea
#PBS -M <email>

### Load modules
module load conda
conda activate ml

## Run
python -u ~/s2s/exploration/plot_predictions.py