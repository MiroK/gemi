# Some simple EMI shapes
import numpy as np


def make_rectangle(fac, x, dx):
    '''(x[0], x[0]+dx[0]) x (x[1], x[1]+dx[1])'''
    assert len(x) == len(dx) == 2
    return fac.addRectangle(x[0], x[1], 0, dx[0], dx[1])


def make_box(fac, x, dx):
    '''(x[0], x[0]+dx[0]) x (x[1], x[1]+dx[1]) x (x[2], x[2]+dx[2])'''    
    assert len(x) == len(dx) == 3
    return fac.addBox(x[0], x[1], x[2], dx[0], dx[1], dx[2])


def make_plus2d(fac, x, dx0, dx1):
    '''Swiss flag where the flat in i-th direction is rectangle with size dxi'''
    assert len(x) == len(dx0) == len(dx1) == 2
    
    shift_x0, shift_y0 = np.array([dx0[0], 0]), np.array([0, dx0[1]])
    shift_x1, shift_y1 = np.array([dx1[0], 0]), np.array([0, dx1[1]])    

    shifts = [shift_x0, -shift_y1, shift_x1, shift_y1, shift_x0, shift_y0,
              -shift_x0, shift_y1, -shift_x1, -shift_y1, -shift_x0]
    # Turtle graphics
    x0 = x
    points = [x0]
    for shift in shifts:
        points.append(points[-1] + shift)
        
    points = [fac.addPoint(*p, z=0) for p in points]
    npts = len(points)
    lines = [fac.addLine(points[i], points[(i+1) % npts]) for i in range(len(points))]
    loop = fac.addCurveLoop(lines)

    return fac.addPlaneSurface([loop])


def make_plus3d(fac, x, dx, sizes):
    '''
    Planes through center with normal in axis direction yield a swiss 2d
    Geometry is defined in terms of shifts of a dx-sized box
    '''
    assert len(x) == len(dx) == len(sizes) == 3
    assert all(dxi > 0 for dxi in dx)
    assert all(size % 2 for size in sizes)
    
    # Basic unit
    box = fac.addBox(x[0], x[1], x[2], dx[0], dx[1], dx[2])
    volumes = [box]
    
    gdim = 3
    for axis in range(gdim):
        size = sizes[axis]
        shift = dx[axis]*np.eye(gdim)[axis]
        if size > 1:
            # Shift in there and back
            steps = size//2
            for k in range(1, 1+steps):
                fac.translate(fac.copy([(gdim, box)]), *(k*shift))
                volumes.append(volumes[-1]+1)
            for k in range(1, 1+steps):
                fac.translate(fac.copy([(gdim, box)]), *(-k*shift))
                volumes.append(volumes[-1]+1)                

    if len(volumes) == 1:
        return box
    
    dimTags = [(3, tag) for tag in volumes]
    obj, *tool = dimTags

    dimTag, _ = fac.fuse([obj], tool)
    # fac.synchronize()

    return dimTag[0][1]

# --------------------------------------------------------------------

if __name__ == '__main__':
    import gmsh

    gmsh.initialize()

    model = gmsh.model
    fac = model.occ

    make_plus3d(fac, x=(0, 0, 0), dx=(1, 0.5, 0.3), sizes=(3, 3, 3))

    fac.synchronize()
    
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    gmsh.finalize()
