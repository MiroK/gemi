# Show making mesh of given characteristic size for the sheet geometry
from gemi.sheet import sheet_geometry
from gemi.cells import make_plus2d
from functools import partial
import gmsh


# Specialize the cell by fixing its size
make_cell = partial(make_plus2d, dx0=(0.2, 0.2), dx1=(0.3, 0.3))
# Silent gmsh and set the mesh size
gmsh.initialize(['', '-v', '0', '-clmax', '0.05'])
model = gmsh.model

# In our geoemtry we want to create sheet with 2 x 4 cells ...
ncells = (2, 4)
# ... that will have the following padding
pads = (0.1, 0.5)

# We get back a lookup table for checing how interfaces are connected to
# cells and extracellular space.
# NOTE: here we set shifts to None so that the cells will be glued to gether.
# Shifts which is larger then the bounding box of the cell allows as to create
# gaps between cells in the sheet
shifts = (1, 1)
model, connectivity = sheet_geometry(model, make_cell=make_cell, ncells=ncells, pads=pads,
                                     shifts=None)

for facet, cells in connectivity.items():
    print(f'Facet {facet} is connected to cells {cells}')

model.occ.synchronize()

# We can checkout the geometry in gmsh
if True:
    gmsh.fltk.initialize()
    gmsh.fltk.run()

from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
# Now we call on to `gmshnics` to get to mesh and mesh functions;
# NOTE: we are meshing a 2-dim problem
nodes, topologies = msh_gmsh_model(model, 2)
mesh, entity_fs = mesh_from_gmsh(nodes, topologies)

# We can dump the mesh for gmsh here (and continue with e.g. meshio)
gmsh.write('demo2d.msh')

# At this point we are done with gmsh
gmsh.finalize()

import dolfin as df
import json

# Just show of dumping to HDF5 ...
with df.HDF5File(mesh.mpi_comm(), 'demo2d.h5', 'w') as out:
    out.write(mesh, 'mesh/')
    out.write(entity_fs[2], 'subdomains/')
    out.write(entity_fs[1], 'interfaces/')        

with open('demo2d.json', 'w') as out:
    json.dump(connectivity, out)

# And Paraview for visual inspection
df.File('demo2d_subdomains.pvd') << entity_fs[2]
df.File('demo2d_interfaces.pvd') << entity_fs[1]
