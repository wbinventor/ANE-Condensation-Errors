import openmc
import openmc.mgxs
from energy_groups import group_structures

scattering = ['anisotropic'] #, 'iso-in-lab']

for scatter in scattering:
    print(scatter)

    directory = '{}/'.format(scatter)

    # Load the last statepoint and summary files
    sp = openmc.StatePoint(directory + 'statepoint.100.h5')

    # Initialize a fine (70-) group MGXS Library for OpenMOC
    mgxs_lib = openmc.mgxs.Library(sp.summary.geometry, by_nuclide=True)
    mgxs_lib.energy_groups = group_structures['CASMO']['70-group']
    mgxs_lib.mgxs_types = ['total', 'nu-fission', 'capture', 'absorption',
                           'consistent nu-scatter matrix', 'chi', 'fission']
    mgxs_lib.correction = None
    mgxs_lib.domain_type = 'cell'
    print('building library')
    mgxs_lib.build_library()
    print('loading from statepoint')
    mgxs_lib.load_from_statepoint(sp)
    print('loaded from statepoint')
    mgxs_lib.dump_to_file(directory=directory)
    print('what is going on')
