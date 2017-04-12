#!/bin/bash
# submit with sbatch run-openmc-iso.slurm

# Number of nodes
#SBATCH -N 16
#SBATCH -n 512
#SBATCH --time=10:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o run-openmc-iso.out

#load default system modules
. /etc/profile.d/modules.sh

cd iso-in-lab
mpirun openmc -s 1
