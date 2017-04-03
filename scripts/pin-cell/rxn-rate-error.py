"""This script generates Table 5.16 and the related analysis. The script
computes U-238 capture and nuclide-integrated absorption rates in energy
ranges A, B and C for different energy group structures. This script is
used to determine how much the U-238 capture resonances contribute to the
negative eigenvalue bias."""


import numpy as np

import openmc.mgxs
import openmoc
from openmoc.opencg_compatible import get_openmoc_geometry
from infermc.energy_groups import group_structures


def get_fluxes(solver, mgxs_lib):
    """

    :param solver:
    :param mgxs_lib:
    :return:
    """

    # Extract the OpenMOC scalar fluxes indexed by FSR, group
    openmoc_fluxes = openmoc.process.get_scalar_fluxes(solver)

    # Extract parameters from OpenMOC geometry to allocate arrays
    openmoc_geometry = solver.getGeometry()
    num_cells = len(mgxs_lib.domains)
    num_fsrs = openmoc_geometry.getNumFSRs()
    num_groups = openmoc_geometry.getNumEnergyGroups()

    # Allocate arrays for ancestor (non-discretized) cell fluxes and volumes
    openmoc_cell_fluxes = np.zeros((num_cells, num_groups), dtype=np.float64)
    openmc_fluxes = np.zeros((num_cells, num_groups), dtype=np.float64)
    volumes = np.zeros(num_cells, dtype=np.float64)
    distances = np.zeros(num_cells, dtype=np.float64)
    fuel_indices = []
    openmc_fiss = 0.

    for i, domain in enumerate(mgxs_lib.domains):

        # Lookup a proxy MGXS to get the flux for this domain (cell)
        mgxs = mgxs_lib.get_mgxs(domain, 'nu-fission')
        mgxs_mean = mgxs.get_xs(nuclides='sum', xs_type='macro')
        flux = mgxs.tallies['flux'].mean.flatten()
        openmc_fluxes[i, :] = flux[::-1]
        openmc_fiss += np.sum(flux[::-1] * mgxs_mean)

        # Get the OpenMC flux in each FSR
        for fsr in range(num_fsrs):

            # Find the OpenMOC cell and its parent for this FSR
            cell = openmoc_geometry.findCellContainingFSR(fsr)
            ancestor = cell.getOldestAncestor()

            # Increment the flux, volume for the ancestor cell for this FSR
            if ancestor.getId() == domain.id:
                fsr_volume = track_generator.getFSRVolume(fsr)
                openmoc_cell_fluxes[i,:] += openmoc_fluxes[fsr,:] * fsr_volume
                volumes[i] += fsr_volume

                centroid = openmoc_geometry.getFSRCentroid(fsr)
                x, y, z = centroid.getX(), centroid.getY(), centroid.getZ()
                distances[i] = np.sqrt(x**2 + y**2 + z**2)

                if ancestor.getName() == 'fuel':
                    fuel_indices.append(i)

    # Divide the fluxes by the ancestor (non-discretized) cell volumes
    openmc_fluxes /= volumes[:,np.newaxis]
    openmc_fluxes /= openmc_fiss
#    openmc_fluxes /= (openmc_fiss * volumes[:,np.newaxis])
    openmoc_fluxes = openmoc_cell_fluxes / volumes[:,np.newaxis]

    fuel_indices = np.unique(fuel_indices)

    return openmc_fluxes, openmoc_fluxes, volumes, distances, fuel_indices


openmoc.log.set_log_level('NORMAL')
opts = openmoc.options.Options()

groups = [1, 2, 4, 8, 16, 25, 40, 70]
scatter = 'iso-in-lab'
num_mesh = 16

# Initialize a fine (70-)group MGXS Library from OpenMC statepoint data
directory = '{}/{}x/'.format(scatter, num_mesh)
sp = openmc.StatePoint(directory + 'statepoint.100.h5')
mgxs_lib = openmc.mgxs.Library.load_from_file(directory=directory)

abs_rel_err = np.zeros((len(groups), 3), dtype=np.float)
capt_rel_err = np.zeros((len(groups), 3), dtype=np.float)
fiss_rel_err = np.zeros((len(groups), 3), dtype=np.float)
u238_frac = np.zeros((len(groups), 3), dtype=np.float)
abs_frac = np.zeros((len(groups), 2), dtype=np.float)

for i, num_groups in enumerate(groups):
    print('# groups = {}'.format(num_groups))

    # Build a coarse group Library from the fine (70-)group Library
    coarse_groups = group_structures['CASMO']['{}-group'.format(num_groups)]
    condense_lib = mgxs_lib.get_condensed_library(coarse_groups)

    # Create an OpenMOC Geometry from the OpenCG Geometry
    openmoc_geometry = get_openmoc_geometry(condense_lib.opencg_geometry)
    openmoc.materialize.load_openmc_mgxs_lib(condense_lib, openmoc_geometry)

    # Discretize the geometry into angular sectors
    cells = openmoc_geometry.getAllMaterialCells()
    for cell_id, cell in cells.items():
        cell.setNumSectors(8)

    # Generate tracks
    track_generator = openmoc.TrackGenerator(openmoc_geometry, 512, 0.001)
    track_generator.setNumThreads(opts.num_omp_threads)
    track_generator.generateTracks(store=False)

    # Instantiate a Solver
    solver = openmoc.CPUSolver(track_generator)
    solver.setNumThreads(opts.num_omp_threads)
    solver.setConvergenceThreshold(1E-7)

    # Run OpenMOC
    solver.computeEigenvalue(opts.max_iters)

    # Extract the normalized fluxes and cell volumes
    openmc_fluxes, openmoc_fluxes, volumes, distances, fuel_indices = \
        get_fluxes(solver, condense_lib)

    # Extract the OpenMC scalar fluxes
    cells = openmoc_geometry.getAllMaterialCells()
    num_cells = len(cells)
    num_groups = openmoc_geometry.getNumEnergyGroups()

    # Allocate arrays for absorption/capture rates by material and energy group
    openmc_abs = np.zeros(num_groups, dtype=np.float)
    openmoc_abs = np.zeros(num_groups, dtype=np.float)
    openmc_capt = np.zeros(num_groups, dtype=np.float)
    openmoc_capt = np.zeros(num_groups, dtype=np.float)
    openmc_fiss = np.zeros(num_groups, dtype=np.float)
    openmoc_fiss = np.zeros(num_groups, dtype=np.float)

    for j, domain in enumerate(condense_lib.domains):

        # Get the capture cross section for this cell
        abs_mgxs = condense_lib.get_mgxs(domain, 'absorption')
        capt_mgxs = condense_lib.get_mgxs(domain, 'capture')
        fiss_mgxs = condense_lib.get_mgxs(domain, 'nu-fission')

        # Compute OpenMC/OpenMOC total capture rates
        abs_mean = abs_mgxs.get_xs(nuclides='sum', xs_type='macro')
        openmc_abs += openmc_fluxes[j,:] * abs_mean.flatten() * volumes[j]
        openmoc_abs += openmoc_fluxes[j,:] * abs_mean.flatten() * volumes[j]

        # Compute OpenMC/OpenMOC total fission rates
        fiss_mean = fiss_mgxs.get_xs(nuclides='sum', xs_type='macro')
        openmc_fiss += openmc_fluxes[j,:] * fiss_mean.flatten() * volumes[j]
        openmoc_fiss += openmoc_fluxes[j,:] * fiss_mean.flatten() * volumes[j]

        # Compute OpenMC/OpenMOC U-238 capture rates
        if domain.name == 'fuel':
            capt_mean = capt_mgxs.get_xs(nuclides=['U-238'], xs_type='macro')
            openmc_capt += openmc_fluxes[j,:] * capt_mean.flatten() * volumes[j]
            openmoc_capt += openmoc_fluxes[j,:] * capt_mean.flatten() * volumes[j]

    print('absorption', np.sum(openmc_abs), np.sum(openmoc_abs))
    print('nu-fission', np.sum(openmc_fiss), np.sum(openmoc_fiss))
    print('keff', np.sum(openmc_fiss) / np.sum(openmc_abs),
          np.sum(openmoc_fiss) / np.sum(openmoc_abs))

    # Find energy group which encompasses 6.67 eV resonance
    min_ind = condense_lib.energy_groups.get_group(6.67e-6) - 1

    # Compute the percent rel. err. in group 27
    abs_rel_err[i,0] = (openmoc_abs[min_ind] - openmc_abs[min_ind]) / openmc_abs[min_ind] * 100.
    capt_rel_err[i,0] = (openmoc_capt[min_ind] - openmc_capt[min_ind]) / openmc_capt[min_ind] * 100
    fiss_rel_err[i,0] = (openmoc_fiss[min_ind] - openmc_fiss[min_ind]) / openmc_fiss[min_ind] * 100

    # Compute the percent rel. err. in all groups
    abs_rel_err[i,2] = (np.sum(openmoc_abs) - np.sum(openmc_abs)) / np.sum(openmc_abs) * 100.
    capt_rel_err[i,2] = (np.sum(openmoc_capt) - np.sum(openmc_capt)) / np.sum(openmc_capt) * 100
    fiss_rel_err[i,2] = (np.sum(openmoc_fiss) - np.sum(openmc_fiss)) / np.sum(openmc_fiss) * 100

    # Compute the percentage of U-238 capture to total absorption in group 27 and all groups
    u238_frac[i,0] = openmc_capt[min_ind] / openmc_abs[min_ind] * 100.
    u238_frac[i,2] = np.sum(openmc_capt) / np.sum(openmc_abs) * 100.

    # Compute the fraction of total absorption in group 27 to all groups
    abs_frac[i,0] = openmc_abs[min_ind] / np.sum(openmc_abs) * 100.

    # Find energy group which encompasses 200 kEV
    max_ind = condense_lib.energy_groups.get_group(2.e-2) - 1

    # Adjust lower energy group if both indices match so that NumPy indexing works
    if min_ind == max_ind:
        min_ind += 1

    # Compute the percent rel. err. in groups 14-27
    abs_rel_err[i,1] = (np.sum(openmoc_abs[max_ind:min_ind]) -
                        np.sum(openmc_abs[max_ind:min_ind])) / \
                       np.sum(openmc_abs[max_ind:min_ind]) * 100.
    capt_rel_err[i,1] = (np.sum(openmoc_capt[max_ind:min_ind]) -
                         np.sum(openmc_capt[max_ind:min_ind])) / \
                        np.sum(openmc_capt[max_ind:min_ind]) * 100
    fiss_rel_err[i,1] = (np.sum(openmoc_fiss[max_ind:min_ind]) -
                         np.sum(openmc_fiss[max_ind:min_ind])) / \
                        np.sum(openmc_fiss[max_ind:min_ind]) * 100

    # Compute the percentage of U-239 capture to total absorption in groups 14-27
    u238_frac[i,1] = np.sum(openmc_capt[max_ind:min_ind]) / \
                     np.sum(openmc_abs[max_ind:min_ind]) * 100.

    # Compute the fraction of total absorption in the resonance range to all groups
    abs_frac[i,1] = np.sum(openmc_abs[max_ind:min_ind]) / \
                    np.sum(openmc_abs) * 100

group_ranges = ['Group 27', 'Groups 14-27', 'All Groups']

# Print U-238 capture and total absorption table for LaTeX
print('Capture and Absorption Rel. Err.')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, group_range in enumerate(group_ranges):
        row += ' {:1.2f} &'.format(capt_rel_err[i,j])
    row += ' &'
    for j, group_range in enumerate(group_ranges):
        row += ' {:1.2f} &'.format(abs_rel_err[i,j])
    print(row[:-1] + '\\\\')

# Print total fission table for LaTeX
print('Fission Rel. Err.')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, group_range in enumerate(group_ranges):
        row += ' {:1.2f} &'.format(fiss_rel_err[i,j])
    print(row[:-1] + '\\\\')

# Print U-238 capture-to-total absorption table for LaTeX
print('U-238 Capture to Total Absorption [%]')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j, group_range in enumerate(group_ranges):
        row += ' {:2.2f} &'.format(u238_frac[i,j])
    print(row[:-1] + '\\\\')

# Print fraction absorption table for LaTeX
print('Absorption Fraction [%]')
for i, num_groups in enumerate(groups):
    row = '{} &'.format(num_groups)
    for j in range(2):
        row += ' {:2.2f} &'.format(abs_frac[i,j])
    print(row[:-1] + '\\\\')
