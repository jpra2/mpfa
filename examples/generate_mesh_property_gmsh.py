import gmsh
import sys
import math
import os
import numpy as np
import pandas as pd


gmsh_elementTypes = {
    1: '2-node line',
    2: '3-node triangle',
    3: '4-node quadrangle',
    4: '4-node tetrahedron',
    5: '8-node hexahedron',
    6: '6-node prism',
    7: '5-node pyramid',
    15:	'1-node point'
}

gmsh.initialize(sys.argv)

gmsh.model.add("test")

# gmsh.open('mesh/cube_structured.msh')
# gmsh.open('mesh/cube_unstructured.msh')
# gmsh.open('mesh/cube_structured2_test.msh')
gmsh.open('mesh/cube10_salome.gmsh')

def get_nodes():
    """return the nodes, center nodes and the boolean boundary_nodes
    """

    all_nodes_in, coords_in, prc_in = gmsh.model.mesh.getNodes(dim=0)
    coords_in = coords_in.reshape((len(all_nodes_in), 3))
    import pdb; pdb.set_trace()

    all_nodes_boundary, coords_boundary_nodes0, prcn = gmsh.model.mesh.getNodesByElementType(3)
    ids_nodes_boundary = np.arange(len(all_nodes_boundary))
    coords_boundary_nodes0 = coords_boundary_nodes0.reshape((len(all_nodes_boundary), 3))

    unique_boundary_nodes = np.unique(all_nodes_boundary)
    coords_boundary_nodes = np.zeros((len(unique_boundary_nodes), 3))

    for i, node in enumerate(unique_boundary_nodes):
        # coords_boundary_nodes[i][:] = gmsh.model.mesh.getNode(node)[0]
        coords = coords_boundary_nodes0[ids_nodes_boundary[all_nodes_boundary==node][0]]
        coords_boundary_nodes[i][:] = coords
    
    import pdb; pdb.set_trace()
    
    
    

elementTypess, elementTag, nodeElement = gmsh.model.mesh.getElements()
print(elementTypess)
import pdb; pdb.set_trace()

# elementTypes = gmsh.model.mesh.getElementTypes()

# elementTags, nodeTags = gmsh.model.mesh.getElementsByType(elementTypes[2])
# print(elementTags)
# print(len(elementTags))

print(gmsh.model.getPhysicalGroups(dim=2))
groups = gmsh.model.getPhysicalGroups(dim=2)


ents = gmsh.model.getEntitiesForPhysicalGroup(2, groups[0][1])
print(ents)
print(gmsh.model.getType(2, ents[0]))
import pdb; pdb.set_trace()

# get_nodes()

import pdb; pdb.set_trace()

import pdb; pdb.set_trace()








import pdb; pdb.set_trace()


