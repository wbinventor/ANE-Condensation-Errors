#!/bin/bash
# submit with sbatch run-openmc-aniso.slurm

# Number of nodes
#SBATCH -N 32
#SBATCH -n 1024
#SBATCH --time=10:00:00
#SBATCH -p sched_mit_nse
#SBATCH -o run-openmc-aniso.out

#load default system modules
. /etc/profile.d/modules.sh

cd anisotropic
mpirun openmc -s 1
