import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

def get_nodes_pressures(weights, faces_pressures, nodes):
    
    nodes_pressures = np.zeros(len(nodes))
    
    for node in nodes:
        test1 = weights['node_index'] == node
        weights_node = weights['weight'][test1]
        faces_node = weights['face_index'][test1]        
        nodes_pressures[node] = np.dot(weights_node, faces_pressures[faces_node])
    
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
    
    
    
    
    
    
    