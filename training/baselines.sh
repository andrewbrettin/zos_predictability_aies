#!/bin/bash
#PBS -N baselines
#PBS -q casper
 
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=01:00:00
#PBS -J 150-180:30

### MESSAGING
#PBS -m ea
#PBS -M <email>

### Variables

### Echo configurations
echo ${PBS_ARRAY_INDEX}


### Load modules
module load conda
conda activate s2s

### Run
printf "\nTRAIN BASELINES\n"
python -u ~/s2s/training/make_baselines.py ${PBS_ARRAY_INDEX}