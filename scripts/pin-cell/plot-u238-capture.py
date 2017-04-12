"""This script is used to generate Figs. 5.3a, 5.4a and 5.5a:

  - the flux in the fuel vs. energy for OpenMC and OpenMOC
  - the OpenMC-OpenMOC flux error vs. energy for innermost/outermost FSRs
  - the OpenMC-OpenMOC flux error vs. FSR for energy ranges A, B and C
1
The MGXS were generated with isotropic in lab scattering and tallied by FSR.
"""

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

# Force non-interactive mode for plotting on clusters
plt.ioff()
sns.set_style('ticks')


def get_capture_rates(mgxs_lib):

    # Extract parameters from OpenMOC geometry to allocate arrays
    num_cells = 16
    num_groups = mgxs_lib.energy_groups.num_groups
    all_capt_rates = np.zeros((num_cells, num_groups), dtype=np.float)

    # Extract U-238 capture rates for each fuel domain
    i = 0
    for domain in mgxs_lib.domains:
        if domain.name == 'fuel':
            capt_mgxs = mgxs_lib.get_mgxs(domain, 'capture')
            capt_rate_tally = capt_mgxs.rxn_rate_tally
            capt_rates = capt_rate_tally.get_values(nuclides=['U238'])
            all_capt_rates[i, :] = capt_rates.flat
            i += 1

    return all_capt_rates


# Query the user for the number of energy groups
scatter = 'iso-in-lab'
num_groups = 70

# Extract the U-238 capture rates
directory = scatter
mgxs_lib = openmc.mgxs.Library.load_from_file(directory=directory)
capt_rates = get_capture_rates(mgxs_lib)

###############################################################################
#          Plot OpenMC-to-OpenMOC Flux Error vs Fuel FSRs in Group 27
###############################################################################

# Plot the U-238 capture rates in group 27
fig = plt.figure()

# Extend the rel er array for matplotlib's step plot of fluxes
capt_rates = sorted(capt_rates[:, 70-27])
capt_rates /= max(capt_rates)
capt_rates = np.insert(capt_rates, 0, capt_rates[0], axis=0)

plt.plot(np.arange(capt_rates.shape[0]), capt_rates,
         drawstyle='steps', color='r', linewidth=3)
plt.xlabel('Fuel FSR', fontsize=12)
plt.ylabel('U-238 Capture Rates', fontsize=12)
plt.savefig('u238-capt-rates-fuel-fsrs.png', bbox_inches='tight')
plt.close()
