"""This script generates Table 5.10 - eigenvalue bias by energy group structure,
FSR discretization, and anisotropic vs. isotropic in lab scattering with MGXS
tallied by FSR."""


import numpy as np

import openmc.mgxs
import openmoc
from openmoc.opencg_compatible import get_openmoc_geometry
from infermc.energy_groups import group_structures


openmoc.log.set_log_level('RESULT')
opts = openmoc.options.Options()

groups = [1, 2, 4, 8, 16, 25, 40, 70]
scattering = ['anisotropic', 'transport', 'iso-in-lab']
num_rings = [1, 2, 4, 8, 16]
keffs = np.zeros((len(scattering), len(groups), len(num_rings)), dtype=np.float)
biases = np.zeros((len(scattering), len(groups), len(num_rings)), dtype=np.float)

for i, scatter in enumerate(scattering):
    print(scatter)

    for j, num_groups in enumerate(groups):
        for k, num_mesh in enumerate(num_rings):
            print('# groups = {}, # mesh = {}'.format(num_groups, num_mesh))

            # Initialize a fine (70-)group MGXS Library from OpenMC statepoint data
            directory = '{}/{}x/'.format(scatter, num_mesh)
            sp = openmc.StatePoint(directory + 'statepoint.100.h5')
            mgxs_lib = openmc.mgxs.Library.load_from_file(directory=directory)

            # Build a coarse group Library from the fine (70-)group Library
            coarse_groups = group_structures['CASMO']['{}-group'.format(num_groups)]
            condense_lib = mgxs_lib.get_condensed_library(coarse_groups)

            # Create an OpenMOC Geometry from the OpenCG Geometry
            openmoc_geometry = get_openmoc_geometry(condense_lib.opencg_geometry)
            openmoc.materialize.load_openmc_mgxs_lib(condense_lib, openmoc_geometry)

            # Apply sector mesh
            cells = openmoc_geometry.getAllMaterialCells()
            for cell_id, cell in cells.items():
                cell.setNumSectors(8)

            # Generate tracks
            track_generator = openmoc.TrackGenerator(openmoc_geometry, 128, 0.01)
            track_generator.setNumThreads(opts.num_omp_threads)
            track_generator.generateTracks()

            # Instantiate a Solver
            solver = openmoc.CPUSolver(track_generator)
            solver.setNumThreads(opts.num_omp_threads)
            solver.setConvergenceThreshold(1E-7)

            # Run OpenMOC
            solver.computeEigenvalue(opts.max_iters)
            keffs[i,j,k] = solver.getKeff()

    # Compute the bias with OpenMC in units of pcm for this scattering type
    biases[i, ...] = (keffs[i, ...] - sp.k_combined[0]) * 1e5

print(biases)

# Print anisotropic table for LaTeX
print('anisotropic')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, num_mesh in enumerate(num_rings):
        row += ' {:1.0f} &'.format(biases[0,i,j])
    print(row[:-1] + '\\\\')

# Print transport table for LaTeX                                             
print('transport')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, num_mesh in enumerate(num_rings):
        row += ' {:1.0f} &'.format(biases[1,i,j])
    print(row[:-1] + '\\\\')

# Print iso-in-lab-table for LaTeX
print('iso-in-lab')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, num_mesh in enumerate(num_rings):
        row += ' {:1.0f} &'.format(biases[2,i,j])
    print(row[:-1] + '\\\\')
