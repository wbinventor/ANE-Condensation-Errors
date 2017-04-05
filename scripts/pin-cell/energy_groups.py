from openmc.mgxs import EnergyGroups
import numpy as np

# Create a global dictionary to store all energy group structures
group_structures = dict()

# Create a sub-dictionary for the CASMO energy group structures
casmo = dict()

# 70-group structure
casmo['70-group'] = EnergyGroups()
group_edges = np.array([0., 0.005e-6, 0.01e-6, 0.015e-6,
                        0.02e-6, 0.025e-6, 0.03e-6, 0.035e-6,
                        0.042e-6, 0.05e-6, 0.058e-6, 0.067e-6,
                        0.08e-6, 0.1e-6, 0.14e-6, 0.18e-6,
                        0.22e-6, 0.25e-6, 0.28e-6, 0.3e-6,
                        0.32e-6, 0.35e-6, 0.4e-6, 0.5e-6,
                        0.625e-6, 0.78e-6, 0.85e-6, 0.91e-6,
                        0.95e-6, 0.972e-6, 0.996e-6, 1.02e-6,
                        1.045e-6, 1.071e-6, 1.097e-6, 1.123e-6,
                        1.15e-6, 1.3e-6, 1.5e-6, 1.855e-6,
                        2.1e-6, 2.6e-6, 3.3e-6, 4.e-6,
                        9.877e-6, 15.968e-6, 27.7e-6, 48.052e-6,
                        75.501e-6, 148.73e-6, 367.26001e-6,
                        906.90002e-6, 1.4251e-3, 2.2395e-3, 3.5191e-3,
                        5.53e-3, 9.118e-3, 15.03e-3, 24.78e-3, 40.85e-3,
                        67.34e-3, 111.e-3, 183.e-3, 302.5e-3, 500.e-3,
                        821.e-3, 1.353, 2.231, 3.679, 6.0655, 20.])
group_edges *= 1e6
casmo['70-group'].group_edges = group_edges


# Store the sub-dictionary in the global dictionary
group_structures['CASMO'] = casmo
