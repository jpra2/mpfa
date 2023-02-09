import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from pack_errors import errors

def get_nodes_pressures(weights, faces_pressures, nodes, dirichlet_boundary, nodes_of_edges):
    
    nodes_pressures = np.zeros(len(nodes))
    
    for node in nodes:
        test1 = weights['node_index'] == node
        weights_node = weights['weight'][test1]
        faces_node = weights['face_index'][test1]        
        nodes_pressures[node] = np.dot(weights_node, faces_pressures[faces_node])
    
    # for boundary in dirichlet_boundary:
    #     edges = boundary['edges']
    #     value = boundary['value']
        
    #     edges_nodes = np.unique(nodes_of_edges[edges].flatten())
    #     nodes_pressures[edges_nodes] = value
        
    return nodes_pressures
    

def get_edge_flux(normal_term, norm_edge_directions, face_pressures, tangent_term, nodes_pressures, faces_adj_by_edges, nodes_of_edges):
    
    
    edges_flux = -normal_term*norm_edge_directions*(face_pressures[faces_adj_by_edges[: ,1]] - face_pressures[faces_adj_by_edges[: ,0]] - tangent_term*(nodes_pressures[nodes_of_edges[:, 1]] - nodes_pressures[nodes_of_edges[:, 0]]))
    
    return edges_flux 

def mount_transmissibility_matrix(normal_term, norm_edge_directions, faces_adj_by_edges, nodes_weights, tangent_term, bool_boundary_edges, edges, faces, nodes_of_edges):
    
    bool_internal_edges = ~bool_boundary_edges
    n_faces = faces.shape[0]
    
    internal_edges = edges[bool_internal_edges]
    
    normal_term_local = -normal_term*norm_edge_directions
    
    b_term = -normal_term_local
    a_term = normal_term_local
    
    nodes_term = -1*normal_term_local*tangent_term
    
    # b1_term = -nodes_term
    # b2_term = nodes_term
    
    nodes_term = np.array([-nodes_term, nodes_term]).T
    
    lines = []
    cols = []
    data = []
    
    for edge in internal_edges:
        nodes_edge = nodes_of_edges[edge]
        faces_index_node_weights = []
        nodes_edge_weights = []
        for node in nodes_edge:
            faces_index_node_weights.append(nodes_weights['face_index'][nodes_weights['node_index'] == node])
            nodes_edge_weights.append(nodes_weights['weight'][nodes_weights['node_index'] == node])
        
        nodes_term_edge = nodes_term[edge]
        
        nodes_term_transm = [nodes_edge_weights[0]*nodes_term_edge[0], nodes_edge_weights[1]*nodes_term_edge[1]]
        a_term_local = a_term[edge]
        b_term_local = b_term[edge]
        faces_adj = faces_adj_by_edges[edge]
        
        data_local = np.concatenate([[b_term_local], [a_term_local], nodes_term_transm[0], nodes_term_transm[1]])
        cols_local = np.concatenate([faces_adj, faces_index_node_weights[0], faces_index_node_weights[1]])
        lines_local = np.repeat(faces_adj[0], len(cols_local))
        
        cols2 = np.concatenate([faces_adj, faces_index_node_weights[0], faces_index_node_weights[1]])
        lines2 = np.repeat(faces_adj[1], len(cols2))
        data_local2 = -data_local
        
        data_local = np.concatenate([data_local, data_local2])
        cols_local = np.concatenate([cols_local, cols2]).astype(np.uint64)
        lines_local = np.concatenate([lines_local, lines2]).astype(np.uint64)
        
        lines.append(lines_local)
        cols.append(cols_local)
        data.append(data_local)
    
    lines = np.concatenate(lines)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    Transmissibility = sp.csc_matrix((data, (lines, cols)), shape=(n_faces, n_faces))
    
    return Transmissibility

def solve_problem(T_matrix, rhs):
    x = linalg.spsolve(T_matrix, rhs)
    return x
 
def get_box(box, centroids):
    
    delta = 0.00001
    p1 = box[0] - delta
    p2 = box[1] + delta    
    
    test1 = (centroids[:, 0] > p1[0]) & (centroids[:, 1] > p1[1]) & (centroids[:, 2] > p1[2])
    
    test2 = (centroids[:, 0] < p2[0]) & (centroids[:, 1] < p2[1]) & (centroids[:, 2] < p2[2])
    
    test3 = test1 & test2
    return test3 
    
   
def define_dirichlet_boundary_edges(edges, nodes_of_edges, nodes_centroids):
    ####
    ### provisorio
    ####
    
    edges_centroid = (nodes_centroids[nodes_of_edges[:, 1]] + nodes_centroids[nodes_of_edges[:, 0]])/2
    
    box1 = np.array([
        [0, 0, 0],
        [0.2, 20, 0]
    ])
    
    edges1 = edges[get_box(box1, edges_centroid)]
    
    box0 = np.array([
        [19.8, 0, 0],
        [20, 20, 0]
    ])
    
    edges0 = edges[get_box(box0, edges_centroid)]
    
    boundary = [
        {
            'edges': edges1,
            'value': 100.0
        },
        {
            'edges': edges0,
            'value': 0.0
        }
    ]
    
    return boundary
    
def update_dirichlet_boundary(dirichlet_boundary, transmissibility, rhs, nodes_of_edges, faces_adj_by_edges, nodes_centroids, kn_edge_face, kt_edge_face, h_distance, faces_centroids, edges_dim):
    
    
    for boundary in dirichlet_boundary:
        edges_diric = boundary['edges']
        value = boundary['value']
        
        for edge in edges_diric:
            face = faces_adj_by_edges[edge][0]
            kn = kn_edge_face[edge][0]
            kt = kt_edge_face[edge][0]
            hd = h_distance[edge][0]
            edge_nodes = nodes_of_edges[edge]
            b1b2 = nodes_centroids[edge_nodes[1]] - nodes_centroids[edge_nodes[0]]
            edge_dim = edges_dim[edge]
            b2b1 = -b1b2
            b1Ob = faces_centroids[face] - nodes_centroids[edge_nodes[0]]
            b2Ob = faces_centroids[face] - nodes_centroids[edge_nodes[1]]
            
            term1 = -(kn)/(hd*edge_dim)
            term2 = np.dot(b2Ob, b2b1)*value
            term3 = np.dot(b1Ob, b1b2)*value
            term4 = -edge_dim**2
            term5 = -(value-value)*kt
            
            term_rhs = -(term1*(term2 + term3) + term5)
            transm_term = (term1 * term4)
            
            transmissibility[face, face] += transm_term
            rhs[face] += term_rhs
    
    # transmissibility.eliminate_zeros()
            
def test_max_limits(min_value, max_value, values):
    test = (values < min_value) | (values > max_value)
    
    if test.sum() > 0:
        raise errors.MinMaxValueTestError
        
def test_xlinearity(faces_centroids: np.ndarray, faces_adj_by_edges: np.ndarray, bool_boundary_edges, faces_pressures, faces):
    
    bool_internal_edges = ~bool_boundary_edges
    
    min_global = faces_centroids.min(axis=0)
    max_global = faces_centroids.max(axis=0)
    
    xmin = min_global[0]
    xmax = max_global[0]
    
    delta = 0.0001
    
    box_min = np.array([
        [xmin-delta, min_global[1], min_global[2]],
        [xmin+delta, max_global[1], max_global[2]]
    ])
    
    faces_min_bool = get_box(box_min, faces_centroids)
    faces_min_centroids = faces_centroids[faces_min_bool][0]
    
    box_max = np.array([
        [xmax-delta, min_global[1], min_global[2]],
        [xmax+delta, max_global[1], max_global[2]]
    ])
    
    faces_max_bool = get_box(box_max, faces_centroids)
    faces_max_centroids = faces_centroids[faces_max_bool][0]
    
    pressure_face_min = faces_pressures[faces_min_bool]
    pressure_face_max = faces_pressures[faces_max_bool]
    
    global_gradient = -(pressure_face_max - pressure_face_min)/(xmax - xmin)
    
    all_values = np.zeros(len(faces_centroids))
    
    other_faces = ~faces_max_bool
    
    gradient = (faces_pressures[other_faces] - pressure_face_max)/(-faces_centroids[other_faces, 0] + xmax)
    
    all_values = np.abs(gradient - global_gradient)
    print('####################################')
    print(f'Max global gradient: {all_values.max()}')
    print('####################################')
    
    local_gradient = np.abs(
        (
            faces_pressures[faces_adj_by_edges[bool_internal_edges, 1]] - 
            faces_pressures[faces_adj_by_edges[bool_internal_edges, 0]])
        )/(
            faces_centroids[faces_adj_by_edges[bool_internal_edges, 1], 0] - 
            faces_centroids[faces_adj_by_edges[bool_internal_edges, 0], 0]
        )
    
    all_values2 = np.abs(np.abs(local_gradient) - np.abs(global_gradient))
        
    # local_gradient = np.abs(local_gradient - global_gradient)
    
    print('####################################')
    print(f'Max local gradient: {all_values2.max()}')
    print('####################################')
    
    
    
    import pdb; pdb.set_trace()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    