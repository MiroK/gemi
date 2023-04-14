from collections import deque, defaultdict
import numpy as np
import itertools
import tqdm, time


def sheet_geometry(model, make_cell, ncells, pads, shifts=None):
    '''
    Use GMSH.OpenCascade functionality to build a sheet of (connected) 
    cells ncells[i] cells in i-th direction. This collection is inclosed 
    in box whose size is determined by padding: pads[i] in i-th direction
    
    Return the model and lookup table of facet tags -> connected cells

    In marking we will have these conventions: extracellular space is tagged
    as 1. Outer boundary of the extracellular space is labeled as

      x[0] == xmin -> 1
      x[0] == xmax -> 2
      x[1] == ymin -> 3
      x[1] == ymax -> 4
      ...

    The EMI cells are >= 2 and their boundaries are >= 2*d + 1,  
    '''
    assert len(ncells) == len(pads)
    assert all(n > 0 for n in ncells)
    assert all(p > 0 for p in pads)
    assert shifts is None or len(shifts) == len(pads)

    gdim = len(ncells)
    # We place the first tile in the origin
    origin = np.zeros(gdim)

    fac = model.occ
    cell0 = make_cell(fac, origin)
    
    xmin, xmax = get_bounding_box(model, gdim)
    # If we do not have the shift for tiling we will compute it from the
    # bounding box of the cell
    if shifts is None:
        shifts = xmax - xmin
    else:
        assert np.all((np.array(shifts) - (xmax-xmin)) > 0), 'Cells would intersect this way'

    shifts = map(np.array, itertools.product(*[shifts[i]*np.arange(ncells[i]) for i in range(gdim)]))
    next(shifts)  # As we do not want to add cell0 again

    cells = [cell0]
    for shift in tqdm.tqdm(shifts, total=np.prod(ncells)-1, desc='Adding cells'):
        x = origin + shift
        cells.append(make_cell(fac, x))

    # With cells inserted we want to get the bounding box to add the extracellular space
    xmin, xmax = get_bounding_box(model, gdim=gdim)
    xmin, xmax = xmin - np.array(pads), xmax + np.array(pads)
    dx = xmax - xmin
    if gdim == 2:
        extrac = fac.addRectangle(xmin[0], xmin[1], 0, dx[0], dx[1])
    else:
        extrac = fac.addBox(xmin[0], xmin[1], xmin[2], dx[0], dx[1], dx[2])
    
    # We compute the intersections of cells with extrac
    print('Fragmenting geometry')
    t0 = time.time()
    _, (extrac, *intrac) = fac.fragment([(gdim, extrac)], [(gdim, cell) for cell in cells],
                                        removeObject=True, removeTool=True)
    fac.synchronize()
    print(f'\tDone fragmenting in {time.time()-t0} s')
    # fac.removeAllDuplicates()
    
    # Now each item in intrac should be one EMI cell
    assert all(len(emi_cell) == 1 for emi_cell in intrac)
    # While extrac (resulting from braeakdown of the obj in fragments has
    # more pieces so not all of it are extrac. Not that (especially in 2d)
    # the extracellular domain may have holes, i.e. consist of several pieces
    emi_cells = set(sum(intrac, []))
    extrac = list(set(extrac) - emi_cells)

    print('Computing cell connectivity')
    t0 = time.time()
    model, connectivity = compute_connectivity(model, fac, xmin, xmax, emi_cells, extrac)
    print(f'\tDone computing connectivity in {time.time()-t0} s')
    return model, connectivity


def compute_connectivity(model, fac, xmin, xmax, emi_cells, extrac_cells):
    '''
    Compute connectivity between EMI cells and cells of extracellular space.
    connectivity is given in terms of ids that represent physical groups 
    of facets and cells. The ids follow convention.
    '''
    assert len(xmin) == len(xmax)
    gdim = len(xmin)
    
    cell_to_facets = {}
    for emi_cell in itertools.chain(emi_cells, extrac_cells):
        _, cell_facets = model.getAdjacencies(*emi_cell)
        cell_to_facets[emi_cell[1]] = cell_facets
    # Now flip it to get facet -> cells
    facet_to_cells = defaultdict(list)
    for cell, facets in cell_to_facets.items():
        consume((facet_to_cells[facet].append(cell) for facet in facets))
        
    extrac_cells = [dimTag[1] for dimTag in extrac_cells]
    emi_cells = [dimTag[1] for dimTag in emi_cells]
    # Gives things physical group id
    cell_id_to_group = {tag: 1 for tag in extrac_cells}
    model.addPhysicalGroup(gdim, extrac_cells, 1)
    # Extrac_Cellsellular are 1 the emi cells are >= 2
    for group, cell_id in enumerate(emi_cells, 2):
        cell_id_to_group[cell_id] = group
        model.addPhysicalGroup(gdim, [cell_id], group)

    # Label outer boundary following the coordinate convention
    boundary_facets = set(facet for facet in facet_to_cells if len(facet_to_cells[facet]) == 1)

    make_center = lambda: 0.5*(xmax + xmin)
    group = 1
    facet_id_to_group = {}
    for dim in range(gdim):
        for x in (xmin, xmax):            
            target = make_center()
            target[dim] = x[dim]
            facet_id = match_facet(fac, boundary_facets, target)
            
            facet_id_to_group[facet_id] = group
            model.addPhysicalGroup(gdim-1, [facet_id], group)
            group += 1            
    # The remaining facets start
    for group, facet_id in enumerate(facet_to_cells.keys()-boundary_facets, group):
        facet_id_to_group[facet_id] = group
        model.addPhysicalGroup(gdim-1, [facet_id], group)

    fac.synchronize()        
    # And remap the connectivity
    connectivity = {}  # Group id of facet -> group ids of connected cells
    for facet_id in facet_to_cells:
        cell_ids = facet_to_cells[facet_id]
        connectivity[facet_id_to_group[facet_id]] = tuple(cell_id_to_group[cell_id] for cell_id in cell_ids)

    return model, connectivity


def get_bounding_box(model, gdim):
    '''From all the points in the model'''
    model.occ.synchronize()

    dimTags = model.getEntities(0)
    vertices = np.array([model.getValue(*dimTag, []) for dimTag in dimTags])

    return np.min(vertices, axis=0)[:gdim], np.max(vertices, axis=0)[:gdim]


def consume(iterator):
    '''Eat it '''
    return deque(iterator, maxlen=0)


def match_facet(factory, candidates, center, tol=1E-13):
    '''Which candidate line match_linees the center'''
    gdim = len(center)
    
    found = False
    for entity in candidates:
        if np.linalg.norm(factory.getCenterOfMass(gdim-1, entity)[:gdim]-np.array(center)) < tol:
            found = True
            break
    assert found

    return entity
