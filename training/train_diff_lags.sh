#!/bin/bash
#PBS -N train
#PBS -q casper
 
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=01:00:00
#PBS -J 60-180:30

### MESSAGING
#PBS -m ea
#PBS -M <email>

### OUTOUT
#PBS -o output/
#PBS -e errors/

### Variables
# TAU=120
TAU=${PBS_ARRAY_INDEX}
ICOORD=4710

### Echo configurations
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}
echo ${TAU}
echo ${PBS_ARRAY_INDEX}


### Load modules
module load conda
conda activate s2s

### Run
printf "\nTRAIN NETWORKS\n"
python -u ~/s2s/full_globe/train.py ${PBS_ARRAY_INDEX} ${ICOORD}

printf "\nPOSTPROCESSING\n"
python -u ~/s2s/full_globe/data_processing/postprocess.py ${PBS_ARRAY_INDEX} ${ICOORD}
