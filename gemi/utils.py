
import numpy as np


def mark_interfaces(facet_f, connectivity, e_tag=None, i_tag=None):
    '''In the new facet_f preserve boundary tagging and only distinguish 
    between interfaces between extecellular space and the cells [2*d + 1]
    and interfaces between cells [2*d + 2]
    '''
    fdim = facet_f.dim()
    mesh = facet_f.mesh()
    # We have a facet function
    assert mesh.topology().dim() - 1 == fdim

    e_interfaces = tuple(k for (k, v) in connectivity.items() if len(v) == 2 and 1 in v)
    i_interfaces = tuple(k for (k, v) in connectivity.items() if len(v) == 2 and 1 not in v)

    new_facet_f = type(facet_f)(mesh, fdim, 0)
    new_values = 1*facet_f.array()

    if e_tag is None:
        e_tag = 2*mesh.geometry().dim() + 1
    if i_tag is None:
        i_tag = e_tag + 1
        
    new_values[np.where(np.isin(new_values, e_interfaces))] = e_tag
    new_values[np.where(np.isin(new_values, i_interfaces))] = i_tag

    new_facet_f.array()[:] = new_values

    return new_facet_f, e_tag, i_tag

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df

    df.parameters['ghost_mode'] = 'shared_facet'
    
    mesh = df.UnitSquareMesh(32, 32)
    
    subdomains = df.MeshFunction('size_t', mesh, 2, 1)
    df.CompiledSubDomain(' && '.join(['(x[0] > 0.25 - DOLFIN_EPS)',
                                      '(x[0] < 0.5 + DOLFIN_EPS)',
                                      '(x[1] > 0.25 - DOLFIN_EPS)',
                                      '(x[1] < 0.75 + DOLFIN_EPS)'])).mark(subdomains, 2)

    df.CompiledSubDomain(' && '.join(['(x[0] < 0.75 + DOLFIN_EPS)',
                                      '(x[0] > 0.5 - DOLFIN_EPS)',
                                      '(x[1] > 0.25 - DOLFIN_EPS)',
                                      '(x[1] < 0.75 + DOLFIN_EPS)'])).mark(subdomains, 3)
                         

    boundaries = df.MeshFunction('size_t', mesh, 1, 0)
    df.CompiledSubDomain('near(x[0], 0)').mark(boundaries, 1)
    df.CompiledSubDomain('near(x[0], 1)').mark(boundaries, 2)
    df.CompiledSubDomain('near(x[1], 0)').mark(boundaries, 3)
    df.CompiledSubDomain('near(x[1], 1)').mark(boundaries, 4)

    df.CompiledSubDomain(' || '.join(
        ['near(x[0], 0.25) && ((x[1] > 0.25 - DOLFIN_EPS) && (x[1] < 0.75 + DOLFIN_EPS))',
         'near(x[1], 0.75) && ((x[0] > 0.25 - DOLFIN_EPS) && (x[0] < 0.5 + DOLFIN_EPS))',
         'near(x[1], 0.25) && ((x[0] > 0.25 - DOLFIN_EPS) && (x[0] < 0.5 + DOLFIN_EPS))'])).mark(boundaries, 5)

    df.CompiledSubDomain(' || '.join(
        ['near(x[0], 0.75) && ((x[1] > 0.25 - DOLFIN_EPS) && (x[1] < 0.75 + DOLFIN_EPS))',
         'near(x[1], 0.75) && ((x[0] > 0.5 - DOLFIN_EPS) && (x[0] < 0.75 + DOLFIN_EPS))',
         'near(x[1], 0.25) && ((x[0] > 0.5 - DOLFIN_EPS) && (x[0] < 0.75 + DOLFIN_EPS))'])).mark(boundaries, 6)

    df.CompiledSubDomain(' || '.join(
        ['near(x[0], 0.5) && ((x[1] > 0.25 - DOLFIN_EPS) && (x[1] < 0.75 + DOLFIN_EPS))'])).mark(boundaries, 7)

    
    connectivity = {1: (1, ),
                    2: (1, ),
                    3: (1, ),
                    4: (1, ),
                    5: (1, 2),
                    6: (1, 3),
                    7: (2, 3)}

    simple, etag, itag = mark_interfaces(boundaries, connectivity)

    # Preserved
    ds0 = df.Measure('ds', domain=mesh, subdomain_data=boundaries)
    ds1 = df.Measure('ds', domain=mesh, subdomain_data=simple)

    f = df.Expression('(x[0]+x[1])*(x[0]+x[1])', degree=2)
    for tag in (1, 2, 3, 4):
        this = df.assemble(f*ds0(tag))
        that = df.assemble(f*ds0(tag))
        assert this > 0
        assert abs(this - that) < 1E-13
        

    dS0 = df.Measure('dS', domain=mesh, subdomain_data=boundaries)
    dS1 = df.Measure('dS', domain=mesh, subdomain_data=simple)
    # Okay intrac-extrac
    this = df.assemble(f*dS0(5) + f*dS0(6))
    that = df.assemble(f*dS1(etag))
    assert this > 0
    assert abs(this - that) < 1E-13
    # Okay intrac-intrac
    this = df.assemble(f*dS0(7))
    that = df.assemble(f*dS1(itag))
    assert this > 0
    assert abs(this - that) < 1E-13
