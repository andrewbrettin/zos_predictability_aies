#!/bin/bash

### JOB NAME
#PBS -N plot

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

### OUTPUT
#PBS -o output/
#PBS -e errors/

## Load modules
module load conda
conda activate ml

## Run
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}

mv errors/${PBS_JOBID}.ER errors/${PBS_JOBNAME}.e${PBS_JOBID}
mv output/${PBS_JOBID}.OU output/${PBS_JOBNAME}.o${PBS_JOBID}

python -u ~/s2s/exploration/plot_confident_preds.py

