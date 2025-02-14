#!/bin/bash

### JOB NAME
#PBS -N template

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=4GB

### RUNTIME
#PBS -l walltime=00:01:00

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
conda activate s2s

## Run
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}

mv errors/${PBS_JOBID}.ER errors/${PBS_JOBNAME}.e${PBS_JOBID}
mv output/${PBS_JOBID}.OU output/${PBS_JOBNAME}.o${PBS_JOBID}

## python -u ~/s2s/data_processing/test.py

