# Show making hierarchy of meshes. The idea here is that it can be quite
# costly to define the model. However, once it is defined we can generate
# several meshes (with different coarsenes) for it
from gemi.sheet import sheet_geometry
from gemi.cells import make_plus3d
from functools import partial
import gmsh

# Specialize the cell by fixing its size
make_cell = partial(make_plus3d, dx=(0.2, 0.2, 0.3), sizes=(3, 3, 3))

gmsh.initialize()
model = gmsh.model

# In our geoemtry we want to create sheet with 2 x 4 cells ...
ncells = (2, 4, 2)
# ... that will have the following padding
pads = (0.1, 0.5, 0.2)

# We get back a lookup table for checing how interfaces are connected to
# cells and extracellular space
model, connectivity = sheet_geometry(model, make_cell=make_cell, ncells=ncells, pads=pads)

model.occ.synchronize()

# We can checkout the geometry in gmsh
if True:
    gmsh.fltk.initialize()
    gmsh.fltk.run()

# Now we call on to `gmshnics` to get to mesh and mesh functions;    
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
# We want to generate several refinements

meshes = []
scales = (1, 0.5, 0.25)
for scale in scales:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', scale)

    # NOTE: we are meshing a 3-dim problem        
    nodes, topologies = msh_gmsh_model(model, 3)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)
    meshes.append(mesh)

    # Ready for next round
    gmsh.model.mesh.clear()


# At this point we are done with gmsh
gmsh.finalize()

for mesh in meshes:
    print(f'Num cells {mesh.num_cells()} with mesh size {mesh.hmin()}')
