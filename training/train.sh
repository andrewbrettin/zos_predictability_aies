#!/bin/bash
### JOB NAME
#PBS -N train

### PROJECT ACCOUNT
 

### RESOURCES 
#PBS -l select=1:ncpus=1:mem=16GB

### RUNTIME
#PBS -l walltime=01:00:00

### QUEUE
#PBS -q casper

### JOB ARRAYS (PBS -J 0-6589)
##########PBS -J 0-6589

### MESSAGING
#PBS -m ea
#PBS -M <email>

### OUTOUT
#PBS -o output/
#PBS -e errors/

### Variables
TAU=10

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
python -u ~/s2s/training/train.py ${TAU} ${PBS_ARRAY_INDEX}

printf "\nPOSTPROCESSING\n"
python -u ~/s2s/training/postprocess.py ${TAU} ${PBS_ARRAY_INDEX}

#### Change error and output filenames
mv errors/${PBS_JOBID}.ER errors/${PBS_JOBNAME}.e${PBS_JOBID}
mv output/${PBS_JOBID}.OU output/${PBS_JOBNAME}.o${PBS_JOBID}

echo "================="
echo "PROCESS COMPLETED"
echo "================="