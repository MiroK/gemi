import pytest

from gemi.sheet import sheet_geometry
from functools import partial
from gemi.cells import make_box, make_plus3d
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import gmsh

import dolfin as df
import numpy as np

df.set_log_level(100)


@pytest.mark.parametrize('shift', [None, (1, 1, 1)])
def test_square_cells(shift):
    # Make the mesh
    make_cell = partial(make_box, dx=(0.25, 0.2, 0.3))
    entity_fs, connectivity = _make_mesh(make_cell, ncells=(2, 3, 2), shifts=shift)

    _marking_3d(cell_f=entity_fs[3], facet_f=entity_fs[2], connectivity=connectivity)

    
@pytest.mark.parametrize('shift', [None, (5, 5, 5)])
def test_plus_cells(shift):
    # Make the mesh
    make_cell = partial(make_plus3d, dx=(0.2, 0.2, 0.3), sizes=(3, 3, 3))
    entity_fs, connectivity = _make_mesh(make_cell, ncells=(3, 3, 2), shifts=shift)

    _marking_3d(cell_f=entity_fs[3], facet_f=entity_fs[2], connectivity=connectivity)
    
# Work horses --------------------------------------------------------
    
def _make_mesh(make_cell, ncells, shifts=None):
    gmsh.initialize()
    model = gmsh.model

    pads = (0.1, 0.5, 0.3)

    model, connectivity = sheet_geometry(model, make_cell=make_cell, ncells=ncells, pads=pads,
                                         shifts=shifts)
    model.occ.synchronize()

    nodes, topologies = msh_gmsh_model(model, 3)
    mesh, entity_fs = mesh_from_gmsh(nodes, topologies)
    gmsh.finalize()
    
    return entity_fs, connectivity


def _marking_3d(cell_f, facet_f, connectivity):
    '''Connectivity is fine'''
    # Each interface is either cell-cell or cell-extracellular
    assert set((1, 2)) == set(map(len, connectivity.values()))

    cell_f = cell_f.array()
    mesh = facet_f.mesh()

    _, f2c = mesh.init(2, 3), mesh.topology()(2, 3)

    # The interface connected cells have the right color
    for facet_tag in connectivity:
        for marked_facet in df.SubsetIterator(facet_f, facet_tag):
            facet_cells = f2c(marked_facet.index())
            assert set(cell_f[facet_cells]) == set(connectivity[facet_tag])

    xmin, ymin, zmin = map(df.Constant, np.min(mesh.coordinates(), axis=0))
    xmax, ymax, zmax = map(df.Constant, np.max(mesh.coordinates(), axis=0))
    # The outer boundaries are marked as promised
    x, y, z = df.SpatialCoordinate(mesh)
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)

    assert abs(df.assemble(df.inner(x - xmin, x - xmin)*ds(1))) < 1E-10
    assert abs(df.assemble(df.inner(x - xmax, x - xmax)*ds(2))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymin, y - ymin)*ds(3))) < 1E-10
    assert abs(df.assemble(df.inner(y - ymax, y - ymax)*ds(4))) < 1E-10
    assert abs(df.assemble(df.inner(z - zmin, z - zmin)*ds(5))) < 1E-10
    assert abs(df.assemble(df.inner(z - zmax, z - zmax)*ds(6))) < 1E-10
    
