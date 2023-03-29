import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty, create_initial_mesh_properties, load_mesh_properties
from datamanager.mesh_data import MeshData
from definitions import defpaths
import numpy as np
import pandas as pd
from mpfa.lsds.gls_weight_2d import CalculateGlsWeight2D, Weight

mesh_path = os.path.join(defpaths.mesh, '2d_unstructured.msh')
mesh_name = 'mpfad_gls_test'
weight_name = 'weights_gls'

###################################################
# mesh_properties = create_initial_mesh_properties(mesh_path, mesh_name)
# mesh_properties.export_data()
####################################################

######################################################
# mesh_properties = MeshProperty()
# mesh_properties.insert_mesh_name([mesh_name])
# mesh_properties.load_data()
# mesh_properties.update_data(
#     {
#      'nodes_centroids': mesh_properties.nodes_centroids[:,0:2]
#      }
# )
# datas_to_rename = {
#     'faces_adj_by_nodes': 'faces_of_nodes',
#     'faces_adj_by_edges': 'adjacencies',
#     'nodes_adj_by_nodes': 'nodes_of_nodes',
#     'edges_adj_by_nodes': 'edges_of_nodes'
# }

# mesh_properties.rename_data(datas_to_rename)

# mesh_properties.export_data()
#################################################

#####################################################
mesh_properties = MeshProperty()
mesh_properties.insert_mesh_name([mesh_name])
mesh_properties.load_data()

weight = Weight()
weight.insert_name(name=weight_name)

calculate_weight = CalculateGlsWeight2D(**mesh_properties.get_all_data())
nodes_weights = calculate_weight.get_weight()
import pdb; pdb.set_trace()


print(mesh_properties)