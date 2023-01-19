import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from definitions import defpaths
from utils import calculate_face_properties
import numpy as np


###
## run the code examples.create_cube_unstructured for generate mesh
###

mesh_path = os.path.join(defpaths.mesh, '2d_square_unstructured.msh')

mesh_create = CreateMeshProperties()
mesh_name = 'square_unstructured_test'
mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
mesh_properties: MeshProperty = mesh_create.create_2d_mesh_data()

faces_area, faces_normal_unitary  = calculate_face_properties.define_normal_and_area(mesh_properties.faces, mesh_properties.nodes_of_faces, mesh_properties.nodes_centroids)

mesh_properties.insert_data(
    {
        'faces_area': faces_area,
        'unitary_normal_faces': faces_normal_unitary
    }
)

calculate_face_properties.correction_faces_vertices_order(mesh_properties.unitary_normal_faces, mesh_properties.nodes_of_faces, np.array([0, 0, 1]))

edges_dim, unitary_normal_edges = calculate_face_properties.create_unitary_normal_edges_xy_plane(mesh_properties.nodes_of_edges, mesh_properties.nodes_centroids, mesh_properties.faces_adj_by_edges, mesh_properties.faces_centroids)

mesh_properties.insert_data(
    {
        'edges_dim': edges_dim,
        'unitary_normal_edges': unitary_normal_edges
    }
)

h_distance = calculate_face_properties.create_face_to_edge_distances(
    faces_centroids=mesh_properties.faces_centroids,
    faces_adj_by_edges=mesh_properties.faces_adj_by_edges,
    nodes_of_edges=mesh_properties.nodes_of_edges,
    edges=mesh_properties.edges,
    nodes_centroids=mesh_properties.nodes_centroids
)



calculate_face_properties.ordenate_nodes_of_edges(
    edges=mesh_properties.edges,
    faces_adj_by_edges=mesh_properties.faces_adj_by_edges,
    nodes_of_faces=mesh_properties.nodes_of_faces,
    nodes_of_edges=mesh_properties.nodes_of_edges
)
import pdb; pdb.set_trace()

mesh_properties.export_data()