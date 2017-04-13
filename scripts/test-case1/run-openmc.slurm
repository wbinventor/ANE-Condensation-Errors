#!/bin/bash
# submit with sbatch run-openmc.slurm

# Number of nodes
#SBATCH -N 32
#SBATCH -n 1024
#SBATCH --time=10:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o run-openmc-iso.out

#load default system modules
. /etc/profile.d/modules.sh

cd iso-in-lab
mpirun openmc -s 1
