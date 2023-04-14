# Geometries for EMI
Use `gmsh` to define geometry for EMI simulations which are commonly formed
by glueing together some characteristic EMI cell. The cells can have gaps
between them filled by extraculllar space or be densely packed. The cells can
be 2d or 2d. A useful functionality is that we labels cells/extracellular spaces
and keep track of their connectivities.

Here is a preview of the functionality taken from `gemi/demo/demo_single_mesh2d.py`
```python
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
```

With the results visualized in Gmsh and paraview
  <p align="center">	    
    <img src="https://github.com/mirok/gemi/blob/master/doc/glued_2d.png">
    <img src="https://github.com/mirok/gemi/blob/master/doc/glued_2d_marking.png">    
  </p>


## Dependencies
- [`gmsh`](https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py)(its Python API)
is used to define the model/geometry. Once meshed, (e.g. through GUI) the mesh
can be saved in `.msh` format and used in some FEM simulators.
- [`gmshnics`](https://github.com/MiroK/gmshnics) is used to alternatively load the
mesh in memory for FEniCS. This then requires FEniCS stack version `2019.1.0` and higher

## Installation
I suppose you want to further develop this in which case I recomment lanching the following
from the source directory.
```bash
pip install -e .
```
## Further usage
For further examples see `/demo` or `/test` folders.
  <p align="center">	    
    <img src="https://github.com/mirok/gemi/blob/master/doc/gap_2d.png">
    <img src="https://github.com/mirok/gemi/blob/master/doc/glued_3d.png">    
  </p>


## TODO
- [ ] More cells (somethign with cylinder)
- [ ] Some example where the boundary cells are different (e.g. without connecting points)
      from the interior cells.
