import numpy as np
import math

def sort_radial_sweep(vs, indices):
    """
    Given a list of vertex positions (vs) and indices
    for verts making up a circular-ish planar polygon,
    returns the vertex indices in order around that poly.
    """
    # indices = np.arange(len(vs))
    assert len(vs) >= 3
    
    # Centroid of verts
    # cent = Vector()
    # for v in vs:
    #     cent += (1/len(vs)) * v
    
    cent = np.mean(vs, axis=0)

    
    # Normalized vector from centroid to first vertex
    # ASSUMES: vs[0] is not located at the centroid
    # r0 = (vs[0] - cent).normalized()
    r0 = (vs[0] - cent)/np.linalg.norm(vs[0] - cent)

    # Normal to plane of poly
    # ASSUMES: cent, vs[0], and vs[1] are not colinear
    # nor = (vs[1] - cent).cross(r0).normalized()
    nor = np.cross(vs[1] - cent, r0)
    nor = nor/np.linalg.norm(nor)

    # Pairs of (vertex index, angle to centroid)
    vpairs = []
    for vi, vpos in zip(indices, vs):
        # r1 = (vpos - cent).normalized()
        r1 = vpos - cent
        r1 = r1/np.linalg.norm(r1)
        dot = np.dot(r1,r0)
        angle = math.acos(max(min(dot, 1), -1))
        # angle *= 1 if nor.dot(r1.cross(r0)) >= 0 else -1
        dottest = np.dot(nor, np.cross(r1, r0))
        angle *= 1 if dottest >= 0 else -1        
        vpairs.append((vi, angle))
    
    # Sort by angle and return indices
    vpairs.sort(key=lambda v: v[1])
    # return np.array([vi for vi, angle in vpairs])
    return np.array([vi for vi, angle in vpairs])


def polygon_area(poly):
    """Calculate the area of 3d polygon from ordered vertices points (poly)

    Args:
        poly (_type_): oredered vertices of polygon. #shape (N, 3)

    Returns:
        area: polygon area
    """
    if isinstance(poly, list):
        poly = np.array(poly)
    #all edges
    edges = poly[1:] - poly[0:1]
    # row wise cross product
    cross_product = np.cross(edges[:-1],edges[1:], axis=1)
    #area of all triangles
    area = np.linalg.norm(cross_product, axis=1)/2
    return sum(area)

def get_unitary_normal_vector(vs):
    """return unitary normal vector

    Args:
        vs (_type_): points of polygon vertices
    """
    cent = np.mean(vs, axis=0)
    v1 = vs[0] - cent
    v2 = vs[1] - cent
    normal = np.cross(v1, v2)
    unitary = normal/np.linalg.norm(normal)
    return unitary
    
    
    
    
