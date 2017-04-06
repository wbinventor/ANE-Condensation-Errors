import os
import re

import numpy as np
import matplotlib

# Force headless backend for plotting on clusters
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

import openmc.mgxs
import openmc.openmoc_compatible
import openmoc
from energy_groups import group_structures
import pyne.ace

# Force non-interactive mode for plotting on clusters
plt.ioff()
sns.set_style('ticks')


def get_fluxes(solver, mgxs_lib, sph):
    """

    :param solver:
    :param mgxs_lib:
    :return:
    """

    # Extract the OpenMOC scalar fluxes indexed by FSR, group
    openmoc_fluxes = openmoc.process.get_scalar_fluxes(solver) * sph

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

    # Reorder SPH factors wrt domains in the same way as the fluxes
    new_sph = np.zeros((num_cells, num_groups), dtype=np.float64)

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

            # FIXME: For non-sectorized case
            sectorized = True
            if ancestor is None:
                ancestor = cell
                sectorized = False

            # Increment the flux, volume for the ancestor cell for this FSR
            if ancestor.getId() == domain.id:
                fsr_volume = track_generator.getFSRVolume(fsr)
                openmoc_cell_fluxes[i,:] += openmoc_fluxes[fsr,:] * fsr_volume
                volumes[i] += fsr_volume
                new_sph[i,:] = sph[fsr,:]

                # FIXME
                if sectorized:
                    centroid = openmoc_geometry.getFSRCentroid(fsr)
                    x, y, z = centroid.getX(), centroid.getY(), centroid.getZ()
                    distances[i] = np.sqrt(x**2 + y**2 + z**2)
                else:
                    centroid = openmoc_geometry.getFSRPoint(fsr)
                    x, y, z = centroid.getX(), centroid.getY(), centroid.getZ()
                    distances[i] = np.sqrt(x**2 + y**2 + z**2)

                if ancestor.getName() == 'fuel':
                    fuel_indices.append(i)

    # Divide the fluxes by the ancestor (non-discretized) cell volumes
    openmc_fluxes /= (openmc_fiss * volumes[:,np.newaxis])
    openmoc_fluxes = openmoc_cell_fluxes / volumes[:,np.newaxis]

    openmc_fluxes = openmc_fluxes[:,::-1]
    openmoc_fluxes = openmoc_fluxes[:,::-1]

    fuel_indices = np.unique(fuel_indices)

    return openmc_fluxes, openmoc_fluxes, volumes, distances, fuel_indices, new_sph


openmoc.log.set_log_level('NORMAL')
opts = openmoc.options.Options()

# Query the user for the number of energy groups
scatter = 'iso-in-lab'
num_groups = 70

directory = '{}/'.format(scatter)

# Create an OpenMOC Geometry from the OpenMC Geometry
mgxs_lib = openmc.mgxs.Library.load_from_file(directory=directory)
openmoc_geometry = \
    openmc.openmoc_compatible.get_openmoc_geometry(mgxs_lib.geometry)

coarse_groups = group_structures['CASMO']['{}-group'.format(num_groups)]
mgxs_lib = mgxs_lib.get_condensed_library(coarse_groups)

# Compute SPH factors
sph, sph_mgxs_lib, sph_indices = \
    openmoc.materialize.compute_sph_factors(
        mgxs_lib, num_azim=512, azim_spacing=0.001, sph_tol=1E-7,
        num_threads=opts.num_omp_threads, max_sph_iters=30)

openmoc_materials = \
    openmoc.materialize.load_openmc_mgxs_lib(sph_mgxs_lib, openmoc_geometry)

np.save('sph.npy', sph)

# Initialize an OpenMOC TrackGenerator and Solver
track_generator = openmoc.TrackGenerator(openmoc_geometry, 512, 0.001)
track_generator.generateTracks(store=False)

# Initialize an OpenMOC Solver
solver = openmoc.CPUSolver(track_generator)
solver.setConvergenceThreshold(1E-7)
solver.setNumThreads(opts.num_omp_threads)

# Run an eigenvalue calulation with the MGXS from OpenMC
solver.computeEigenvalue(opts.max_iters)
solver.printTimerReport()


###############################################################################
#                         Extracting Scalar Fluxes
###############################################################################

openmoc.log.py_printf('NORMAL', 'Plotting data...')

# Extract the normalized fluxes and cell volumes
openmc_fluxes, openmoc_fluxes, volumes, distances, fuel_indices, new_sph = \
    get_fluxes(solver, mgxs_lib, sph)

# Extract energy group edges
group_edges = mgxs_lib.energy_groups.group_edges
#group_edges *= 1e6      # Convert to units of eV
group_edges[0] = 1e-5   # Adjust lower bound (for loglog scaling)

# Extend the mgxs values array for matplotlib's step plot of fluxes
openmc_fluxes = np.insert(openmc_fluxes, 0, openmc_fluxes[:,0], axis=1)
openmoc_fluxes = np.insert(openmoc_fluxes, 0, openmoc_fluxes[:,0], axis=1)


###############################################################################
#              Compute OpenMC-to-OpenMOC Scalar Flux Errors
###############################################################################

# Instantiate a PyNE ACE continuous-energy cross sections library
xs_dir = os.environ['OPENMC_CROSS_SECTIONS']
xs_dir = re.sub('cross_sections.xml$', '', xs_dir)
pyne_lib = pyne.ace.Library(xs_dir + '293.6K/U_238_293.6K.ace')
pyne_lib.read('92238.71c')

# Extract the U-235 data from the library
u238 = pyne_lib.tables['92238.71c']

# Extract the continuous-energy U-238 fission cross section data
capture = u238.reactions[102]

# Compute the percent relative error in the flux
delta_flux = openmoc_fluxes - openmc_fluxes
rel_err = delta_flux / openmc_fluxes * 100.


###############################################################################
# Load Relative Errors w/o SPH From NumPy Data Store
###############################################################################

rel_err_no_sph = np.load('rel-err-no-sph.npy')


###############################################################################
#        Plot OpenMC-to-OpenMOC Innermost/Outermost Scalar Flux Error
###############################################################################

min_fsr = fuel_indices[-1]
max_fsr = fuel_indices[0]

# Plot the error for the innermost and outermost FSRS atop each other
fig, ax1 = plt.subplots()

plt.plot(group_edges, rel_err[min_fsr,:],
         drawstyle='steps', color='b', linestyle='-', linewidth=2)
plt.plot(group_edges, rel_err_no_sph[min_fsr,:], alpha=0.5,
         drawstyle='steps', color='b', linestyle='--', linewidth=2)
plt.plot(group_edges, rel_err[max_fsr,:],
         drawstyle='steps', color='r', linestyle='-', linewidth=2)
plt.plot(group_edges, rel_err_no_sph[max_fsr,:], alpha=0.5,
         drawstyle='steps', color='r', linestyle='--', linewidth=2)
plt.plot(group_edges, np.nanmean(rel_err[min(min_fsr, max_fsr):max(min_fsr, max_fsr)+1,:], axis=0),
         drawstyle='steps', color='darkorange', linestyle='-', linewidth=2)
plt.plot(group_edges, np.nanmean(rel_err_no_sph[min(min_fsr, max_fsr):max(min_fsr, max_fsr)+1,:], axis=0), alpha=0.5,
         drawstyle='steps', color='darkorange', linestyle='--', linewidth=2)

plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Relative Error [%]', fontsize=12)
plt.xlim((min(group_edges), max(group_edges)))
plt.xscale('log')
plt.legend(['Inner (SPH)', 'Inner (No SPH)',
            'Outer (SPH)', 'Outer (No SPH)',
            'All (SPH)', 'All (No SPH)'], fontsize=12)

# Create loglog plot of U-238 continuous-energy capture cross section
ax2 = ax1.twinx()
ax2.loglog(u238.energy*1e6, capture.sigma, color='g', \
           linewidth=1, zorder=1)
ax2.set_ylabel('U-238 Capture XS [barns]', color='g', fontsize=12)
ax2.set_yscale('log')

ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
ax1.patch.set_visible(False)       # hide the 'canvas'

plt.savefig('rel-err-inner-outer-sph.png', bbox_inches='tight')
plt.close()


###############################################################################
#  Plot OpenMC-to-OpenMOC Innermost/Outermost SPH Factors
###############################################################################

# Reverse the SPH factors in energy for plotting
new_sph = new_sph[:,::-1]

# Extend the mgxs values array for matplotlib's step plot of fluxes
new_sph = np.insert(new_sph, 0, new_sph[:,0], axis=1)

# Plot the error for the innermost and outermost FSRS atop each other
fig, ax1 = plt.subplots()

plt.plot(group_edges, new_sph[min_fsr,:],
         drawstyle='steps', color='b', linewidth=2)
plt.plot(group_edges, new_sph[max_fsr,:],
         drawstyle='steps', color='r', linewidth=2)

plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('SPH Factor', fontsize=12)
plt.xlim((min(group_edges), max(group_edges)))
plt.xscale('log')
plt.legend(['Inner', 'Outer'], fontsize=12, loc=2)

# Create loglog plot of U-238 continuous-energy capture cross section
ax2 = ax1.twinx()
ax2.loglog(u238.energy*1e6, capture.sigma, color='g', \
           linewidth=1, zorder=1)
ax2.set_ylabel('U-238 Capture XS [barns]', color='g', fontsize=12)
ax2.set_yscale('log')

ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
ax1.patch.set_visible(False) # hide the 'canvas'
ax1.get_yaxis().get_major_formatter().set_useOffset(False)

plt.savefig('sph-inner-outer.png', bbox_inches='tight')
plt.close()


###############################################################################
#              Plot SPH Factors vs. Fuel FSRs in Group 27
###############################################################################

# Plot the SPH factors for group 27
fig = plt.figure()

# Array index for group 27 with the U-238 capture resonance at 6.67ev
group_index = 70 - 27 + 1

print(new_sph[:, group_index])

# Extend the sph array for matplotlib's step plot of fluxes
hmm_indices = fuel_indices[::-1]
new_sph = new_sph[hmm_indices, group_index]
new_sph = np.insert(new_sph, 0, new_sph[0], axis=0)
#sph = sph[sph_indices, group_index]
#sph = np.insert(sph, 0, sph[0], axis=0)

print('sph', new_sph)

plt.plot(np.arange(new_sph.shape[0]), new_sph, drawstyle='steps', color='b', linewidth=3)
plt.xlabel('Fuel FSR', fontsize=12)
plt.ylabel('SPH Factor (Group 27)', fontsize=12)
plt.xlim((0, new_sph.shape[0]-1))
plt.savefig('sph-fuel-fsrs.png', bbox_inches='tight')
plt.close()


###############################################################################
#        Plot OpenMC-to-OpenMOC Flux Error Across Fuel FSRs in Group 27
###############################################################################

# Plot the relative error for group 27
fig = plt.figure()

# Array index for group 27 with the U-238 capture resonance at 6.67ev
group_index = 70 - 27 + 1

print(rel_err[:, group_index])

# Extend the rel er array for matplotlib's step plot of fluxes
fuel_indices = fuel_indices[::-1]
rel_err = rel_err[fuel_indices, group_index]
rel_err = np.insert(rel_err, 0, rel_err[0], axis=0)

print(fuel_indices)

plt.plot(np.arange(rel_err.shape[0]), rel_err, drawstyle='steps', color='b', linewidth=3)
plt.xlabel('Fuel FSR', fontsize=12)
plt.ylabel('Relative Error [%]', fontsize=12)
plt.xlim((0, rel_err.shape[0]-1))
plt.savefig('rel-err-fuel-fsrs-sph.png', bbox_inches='tight')
plt.close()
