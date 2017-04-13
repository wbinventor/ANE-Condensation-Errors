#!/bin/bash
# submit with sbatch sph-aniso.slurm

# Number of nodes
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=05:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o sph-aniso.out

#load default system modules
. /etc/profile.d/modules.sh

python sph-factors.py
