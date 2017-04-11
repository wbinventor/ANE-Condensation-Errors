#!/bin/bash
# submit with sbatch sph-factors.slurm

# Number of nodes
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=10:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o sph-factors-%j.out

#load default system modules
. /etc/profile.d/modules.sh

python sph-factors.py
