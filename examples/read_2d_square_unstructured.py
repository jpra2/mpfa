import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from definitions import defpaths
from utils import calculate_face_properties
import numpy as np
from mpfa.linearity_preserving.preprocess import MpfaLinearityPreservingPreprocess2D

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

edges_dim, unitary_normal_edges = calculate_face_properties.create_unitary_normal_edges_xy_plane(mesh_properties.nodes_of_edges, mesh_properties.nodes_centroids, mesh_properties.faces_adj_by_edges, mesh_properties.faces_centroids, mesh_properties.bool_boundary_edges)

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
    nodes_centroids=mesh_properties.nodes_centroids,
    bool_boundary_edges=mesh_properties.bool_boundary_edges
)

calculate_face_properties.ordenate_nodes_of_edges(
    edges=mesh_properties.edges,
    faces_adj_by_edges=mesh_properties.faces_adj_by_edges,
    nodes_of_faces=mesh_properties.nodes_of_faces,
    nodes_of_edges=mesh_properties.nodes_of_edges
)

mpfaprepropcess = MpfaLinearityPreservingPreprocess2D()



tk_points = mpfaprepropcess.create_Tk_point(
    mesh_properties.nodes_centroids[mesh_properties.nodes_of_edges]
)

phis_and_thethas = mpfaprepropcess.create_phis_and_thetas(
    mesh_properties.nodes,
    mesh_properties.edges,
    mesh_properties.faces,
    mesh_properties.nodes_centroids,
    mesh_properties.faces_centroids,
    mesh_properties.nodes_of_edges,
    tk_points,
    mesh_properties.edges_of_faces,
    mesh_properties.faces_adj_by_nodes,
    mesh_properties.edges_adj_by_nodes,
    mesh_properties.faces_adj_by_edges
)

tk_ok = mpfaprepropcess.create_Tk_Ok_vector(
    mesh_properties.faces_adj_by_edges,
    tk_points,
    mesh_properties.faces_centroids,
    mesh_properties.bool_boundary_edges,
    mesh_properties.edges
)

permeability = np.zeros((len(mesh_properties.faces), 2, 2))
permeability[:,0,0] = 1
permeability[:,1,1] = 2

kn_kt_tk_ok = mpfaprepropcess.create_kn_and_kt_Tk_Ok(
    tk_ok['tk_ok_vector'],
    tk_ok,
    permeability
)

q0_tk = mpfaprepropcess.create_q0_tk_vector(mesh_properties.edges_adj_by_nodes, tk_points, mesh_properties.nodes_centroids)

neta_kn_kt_q0_tk =  mpfaprepropcess.create_neta_kn_and_kt_Q0_Tk(q0_tk['q0_tk_vector'], q0_tk, mesh_properties.faces_adj_by_edges, permeability, mesh_properties.bool_boundary_edges, mesh_properties.edges, h_distance)

lambda_values = mpfaprepropcess.create_weights(
    kn_kt_tk_ok, 
    neta_kn_kt_q0_tk, 
    phis_and_thethas, 
    mesh_properties.edges_adj_by_nodes, 
    mesh_properties.faces_adj_by_nodes, 
    mesh_properties.bool_boundary_nodes, 
    mesh_properties.nodes,
    mesh_properties.faces_adj_by_edges,
    mesh_properties.edges,
    mesh_properties.bool_boundary_edges,
    mesh_properties.edges_of_faces
)

kn_edge_face, kt_edge_face = mpfaprepropcess.create_edge_face_kn_kt(
    mesh_properties.nodes_of_edges,
    mesh_properties.nodes_centroids,
    mesh_properties.faces_adj_by_edges,
    permeability,
    mesh_properties.bool_boundary_edges
)

normal_term = mpfaprepropcess.get_normal_term(kn_edge_face, h_distance, mesh_properties.bool_boundary_edges)

tangent_term = mpfaprepropcess.get_tangent_term(
    mesh_properties.nodes_of_edges,
    mesh_properties.nodes_centroids,
    mesh_properties.faces_adj_by_edges,
    mesh_properties.faces_centroids,
    kn_edge_face,
    kt_edge_face,
    h_distance,
    mesh_properties.bool_boundary_edges
)

from mpfa.linearity_preserving import equations

transmissibility = equations.mount_transmissibility_matrix(
    normal_term,
    mesh_properties.edges_dim,
    mesh_properties.faces_adj_by_edges,
    lambda_values,
    tangent_term,
    mesh_properties.bool_boundary_edges,
    mesh_properties.edges,
    mesh_properties.faces,
    mesh_properties.nodes_of_edges
)

dist1 = np.linalg.norm(mesh_properties.faces_centroids - np.array([0, 0, 0]), axis=1)
dist2 = np.linalg.norm(mesh_properties.faces_centroids - np.array([20, 20, 0]), axis=1)

test1 = dist1 <= dist1.min()
test2 = dist2 <= dist2.min()

face1 = mesh_properties.faces[test1][0]
face0 = mesh_properties.faces[test2][0]

transmissibility[face1,:] = 0
transmissibility[face0,:] = 0
transmissibility[face1, face1] = 1
transmissibility[face0, face0] = 1
transmissibility.eliminate_zeros()


rhs = np.zeros(transmissibility.shape[0])
rhs[face1] = 1

from mpfa.linearity_preserving import equations

pressure = equations.solve_problem(transmissibility, rhs)

nodes_pressures = equations.get_nodes_pressures(lambda_values, pressure, mesh_properties.nodes)

edges_flux = equations.get_edge_flux(normal_term, mesh_properties.edges_dim, pressure, tangent_term, nodes_pressures, mesh_properties.faces_adj_by_edges, mesh_properties.nodes_of_edges)

bool_internal_edges = ~mesh_properties.bool_boundary_edges
faces_adj = mesh_properties.faces_adj_by_edges

rhs2 = np.bincount(
    np.concatenate([faces_adj[bool_internal_edges, 0], faces_adj[bool_internal_edges, 1]]).astype(np.int64),
    weights=np.concatenate([edges_flux[bool_internal_edges], -edges_flux[bool_internal_edges]])
)



import pdb; pdb.set_trace()




import pdb; pdb.set_trace()

mesh_properties.export_data()