#!/bin/bash
# submit with sbatch plot-flux-sph.slurm

# Number of nodes
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o plot-flux-sph.out

#load default system modules
. /etc/profile.d/modules.sh

python plot-flux-sph.py
