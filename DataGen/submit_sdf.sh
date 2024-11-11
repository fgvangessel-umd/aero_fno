#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -A fuge-prj-jrl

module load singularity
singularity shell /home/fvangess/scratch/singularity_files/modulus.sif

singularity exec --nv \
        /home/fvangess/scratch/singularity_files/modulus.sif \
        bash -c 'python sdf_data.py' 2>&1 | tee out.txt
