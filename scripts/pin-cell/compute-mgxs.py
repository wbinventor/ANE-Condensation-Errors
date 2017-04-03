import openmc
import openmc.mgxs
from infermc.energy_groups import group_structures

scattering = ['iso-in-lab', 'anisotropic', 'transport', 'iso-in-lab']
num_rings = [1, 2, 4, 8, 16,]

for scatter in scattering:
    print(scatter)
    for rings in num_rings:
        print('# rings: {}'.format(rings))

        directory = '{}/{}x/'.format(scatter, rings)

        # Load the last statepoint and summary files
        sp = openmc.StatePoint(directory + 'statepoint.100.h5')

        # Initialize a fine (70-) group MGXS Library for OpenMOC
        mgxs_lib = openmc.mgxs.Library(sp.summary.openmc_geometry, by_nuclide=True)
        mgxs_lib.energy_groups = group_structures['CASMO']['70-group']
        if scatter == 'transport':
            mgxs_lib.mgxs_types = ['nu-transport', 'nu-fission', 'nu-scatter matrix', 'chi', 'fission', 'capture', 'absorption']
            mgxs_lib.correction = 'P0'
        else:
            mgxs_lib.mgxs_types = ['total', 'nu-fission', 'nu-scatter matrix', 'chi', 'fission', 'capture', 'absorption']
            mgxs_lib.correction = None
        mgxs_lib.domain_type = 'cell'
        mgxs_lib.build_library()
        mgxs_lib.load_from_statepoint(sp)
        mgxs_lib.dump_to_file(directory=directory)
