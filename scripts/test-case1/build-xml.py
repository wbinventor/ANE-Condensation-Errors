import os
import glob
import copy

import openmc
import openmc.mgxs
import openmc.openmoc_compatible
from energy_groups import group_structures

##################   Exporting to OpenMC materials.xml File  ###################

# Instantiate some Materials and register the appropriate Nuclides
uo2 = openmc.Material(name='UO2 Fuel')
uo2.set_density('g/cm3', 10.29769)
uo2.add_nuclide('U235', 5.5815e-4)
uo2.add_nuclide('U238', 2.2408e-2)
uo2.add_nuclide('O16', 4.5829e-2)

helium = openmc.Material(name='Helium')
helium.set_density('g/cm3', 0.001598)
helium.add_nuclide('He4', 2.4044e-4)

zircaloy = openmc.Material(name='Zircaloy 4')
zircaloy.set_density('g/cm3', 6.55)
zircaloy.add_nuclide('O16', 3.0743e-4)
zircaloy.add_nuclide('Fe56', 1.3610e-4)
zircaloy.add_nuclide('Zr90', 2.1827e-2)

borated_water = openmc.Material(name='Borated Water')
borated_water.set_density('g/cm3', 0.740582)
borated_water.add_nuclide('B10', 8.0042e-6)
borated_water.add_nuclide('B11', 3.2218e-5)
borated_water.add_nuclide('H1', 4.9457e-2)
borated_water.add_nuclide('O16', 2.4672e-2)
borated_water.add_s_alpha_beta('HH2O')

# Instantiate a MaterialsFile, register all Materials, and export to XML
materials_file = openmc.Materials([uo2, helium, zircaloy, borated_water])
materials_file.default_xs = '71c'


###################   Exporting to OpenMC geometry.xml File  ###################

# Instantiate ZCylinder surfaces
fuel_or = openmc.ZCylinder(x0=0, y0=0, R=0.39218, name='Fuel OR')
clad_ir = openmc.ZCylinder(x0=0, y0=0, R=0.40005, name='Clad IR')
clad_or = openmc.ZCylinder(x0=0, y0=0, R=0.45720, name='Clad OR')
min_x = openmc.XPlane(x0=-0.62992, name='min x')
max_x = openmc.XPlane(x0=+0.62992, name='max x')
min_y = openmc.YPlane(y0=-0.62992, name='min y')
max_y = openmc.YPlane(y0=+0.62992, name='max y')
min_z = openmc.ZPlane(z0=-0.62992, name='min z')
max_z = openmc.ZPlane(z0=+0.62992, name='max z')

min_x.boundary_type = 'reflective'
max_x.boundary_type = 'reflective'
min_y.boundary_type = 'reflective'
max_y.boundary_type = 'reflective'
min_z.boundary_type = 'reflective'
max_z.boundary_type = 'reflective'

# Instantiate Cells
fuel = openmc.Cell(name='fuel')
gap = openmc.Cell(name='gap')
clad = openmc.Cell(name='clad')
water = openmc.Cell(name='water')
root_cell = openmc.Cell(name='root')

# Use surface half-spaces to define regions
fuel.region = -fuel_or
gap.region = +fuel_or & -clad_ir
clad.region = +clad_ir & -clad_or
water.region = +clad_or
root_cell.region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z

# Instantiate Universe
pin = openmc.Universe(name='pin cell universe')
root_univ = openmc.Universe(universe_id=0, name='root universe')

# Register fills with Cells
fuel.fill = uo2
gap.fill = helium
clad.fill = zircaloy
water.fill = borated_water
root_cell.fill = pin

# Register Cells with Universe
pin.add_cells([fuel, gap, clad, water])
root_univ.add_cell(root_cell)

# Instantiate a Geometry and register the root Universe
geometry = openmc.Geometry()
geometry.root_universe = root_univ

# Mesh the fuel and water with rings
openmoc_geometry = \
    openmc.openmoc_compatible.get_openmoc_geometry(geometry)
all_cells = openmoc_geometry.getAllMaterialCells()
all_cells[fuel.id].setNumRings(16)
all_cells[water.id].setNumRings(16)

# FIXME
#all_cells[water.id].setNumSectors(8)
#all_cells[fuel.id].setNumSectors(8)
#all_cells[clad.id].setNumSectors(8)
#all_cells[gap.id].setNumSectors(8)

openmoc_geometry.subdivideCells()
openmc_geometry = \
    openmc.openmoc_compatible.get_openmc_geometry(openmoc_geometry)


###################   Exporting to OpenMC settings.xml File  ###################

# Construct uniform initial source distribution over fissionable zones
lower_left =  [-0.62992, -0.62992, -0.62992]
upper_right = [++0.62992, +0.62992, +0.62992]
source = openmc.source.Source(space=openmc.stats.Box(lower_left, upper_right))
source.space.only_fissionable = True

# Instantiate a SettingsFile
settings_file = openmc.Settings()
settings_file.batches = 100
settings_file.inactive = 10
settings_file.particles = 100000000
settings_file.output = {'tallies': False}
settings_file.source = source
settings_file.sourcepoint_write = False


####################   Exporting to OpenMC plots.xml File  #####################

plot = openmc.Plot(plot_id=1)
plot.width = [0.62992 * 2] * 2
plots_file = openmc.Plots([plot])


######################   Move Files into Directories  #########################

scattering = ['anisotropic', 'iso-in-lab']

for scatter in scattering:
    print(scatter)

    if scatter == 'iso-in-lab':
        materials_file.make_isotropic_in_lab()

    materials_file.export_to_xml()
    openmc_geometry.export_to_xml()
    settings_file.export_to_xml()
    plots_file.export_to_xml()

    # Initialize a fine (70-) group MGXS Library for OpenMOC
    mgxs_lib = openmc.mgxs.Library(openmc_geometry, by_nuclide=True)
    mgxs_lib.energy_groups = group_structures['CASMO']['70-group']
    mgxs_lib.mgxs_types = ['total', 'nu-fission',
                           'consistent nu-scatter matrix', 'chi',
                           'fission', 'capture', 'absorption']
    mgxs_lib.correction = None
    mgxs_lib.domain_type = 'cell'
    mgxs_lib.build_library()

    # Create a "tallies.xml" file for the MGXS Library
    tallies_file = openmc.Tallies()
    mgxs_lib.add_to_tallies_file(tallies_file, merge=True)
    tallies_file.export_to_xml()

    # Move files
    for xml_file in glob.glob('*.xml'):
        if not os.path.exists('{}'.format(scatter)):
            os.makedirs('{}'.format(scatter))
        os.rename(xml_file, '{}/{}'.format(scatter, xml_file))
