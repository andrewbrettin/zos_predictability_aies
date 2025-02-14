#!/bin/bash
#PBS -N tensors
#PBS -q casper
 
#PBS -l select=1:ncpus=2:mem=64GB
#PBS -l walltime=04:00:00
#PBS -m ea
#PBS -M <email>
#PBS -J 150-180:30

### Variables
TAU=${PBS_ARRAY_INDEX}

# ### Echo configurations
echo ${PBS_JOBNAME}
echo ${PBS_JOBID}


### Load modules
module load conda
conda activate s2s

### Run
python -u ~/s2s/data_processing/make_tensors.py ${TAU}
python -u ~/s2s/data_processing/save_icoords.py ${TAU}