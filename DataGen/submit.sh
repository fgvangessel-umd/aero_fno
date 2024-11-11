#!/bin/bash

#SBATCH -t 00:05:00
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -A fuge-prj-jrl
#SBATCH --gpus=a100_1g.5gb:1

module load singularity

singularity exec --nv \
        /home/fvangess/scratch/singularity_files/modulus.sif \
        bash -c 'python test_mpi.py' 2>&1 | tee out.txt