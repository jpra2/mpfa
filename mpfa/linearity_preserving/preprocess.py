import numpy as np
import pandas as pd

class MpfaLinearityPreservingPreprocess2D:
    
    def create_Tk_point(self, centroids_of_edges_nodes, tal=0.5):
        
        assert (tal > 0) and (tal < 1)
        edges_points = centroids_of_edges_nodes
        Tk = edges_points[:,0]*tal +(1-tal)*edges_points[:, 1]
        
        return Tk
    
    def create_phis_and_thetas(self, nodes, edges, faces, nodes_centroids, faces_centroids, nodes_of_edges, tk_points, edges_of_faces, faces_adj_by_nodes, edges_adj_by_nodes):
        """create phis and thetas

        Args:
            nodes (_type_): _description_
            edges (_type_): _description_
            faces (_type_): _description_
            nodes_centroids (_type_): _description_
            faces_centroids (_type_): _description_
            nodes_of_edges (_type_): _description_
            tk_points (_type_): _description_
            edges_of_faces (_type_): _description_

        Returns:
            _type_: dict where key = tuple(node, face), value = phi1, phi2, theta1, theta2
        """
        
        index_tuples = []
        phis_and_thetas = []
        edges_tuples = []
        
        dtype_phis_and_thetas = [('phi1', float), ('phi2', float), ('theta1', float), ('theta2', float), ('edge1', np.uint64), ('edge2', np.uint64), ('node_index', np.uint64), ('face_index', np.uint64)]
        
        for node in nodes:
            faces_around_node = faces_adj_by_nodes[node]
            edges_around_node = edges_adj_by_nodes[node]
            node_centroid = nodes_centroids[node]
            
            for face in faces_around_node:
                index_tuples.append((node, face))
                faces_edges = edges_of_faces[face]
                intersect_edges = np.intersect1d(edges_around_node, faces_edges)
                position1 = np.argwhere(edges_around_node == intersect_edges[0])
                position2 = np.argwhere(edges_around_node == intersect_edges[1])
                if position2 > position1:
                    pass
                else:
                    intersect_edges[:] = np.flip(intersect_edges)
                    
                
                tk_edges = tk_points[intersect_edges]
                face_centroid = faces_centroids[face]
                
                q0ok = np.linalg.norm(face_centroid - node_centroid)
                
                q0tk = np.linalg.norm(tk_edges[0] - node_centroid)
                oktk = np.linalg.norm(face_centroid - tk_edges[0])
                
                cos_theta1 = (oktk**2 - (q0ok**2 + q0tk**2))/(-2*q0ok*q0tk)
                cos_phi1 =   (q0tk**2 - (q0ok**2 + oktk**2))/(-2*q0ok*oktk)
                
                q0tk1 = np.linalg.norm(tk_edges[1] - node_centroid)
                oktk1 = np.linalg.norm(face_centroid - tk_edges[1])
                
                cos_theta2 = (oktk1**2 - (q0ok**2 + q0tk1**2))/(-2*q0ok*q0tk1)
                cos_phi2 =   (q0tk1**2 - (q0ok**2 + oktk1**2))/(-2*q0ok*oktk1)
                
                theta1 = np.arccos(cos_theta1)
                phi1 = np.arccos(cos_phi1)
                theta2 = np.arccos(cos_theta2)
                phi2 = np.arccos(cos_phi2)
                
                resp = np.array([phi1, phi2, theta1, theta2])
                
                edges_tuples.append(intersect_edges)
                phis_and_thetas.append(resp)
        
        phis_and_thetas = np.array(phis_and_thetas)
        edges_tuples = np.array(edges_tuples)
        index_tuples = np.array(index_tuples)
        
        array_structured = np.zeros(len(phis_and_thetas), dtype=dtype_phis_and_thetas)
        
        array_structured['phi1'] = phis_and_thetas[:, 0]
        array_structured['phi2'] = phis_and_thetas[:, 1]
        array_structured['theta1'] = phis_and_thetas[:, 2]
        array_structured['theta2'] = phis_and_thetas[:, 3]
        array_structured['edge1'] = edges_tuples[:, 0]
        array_structured['edge2'] = edges_tuples[:, 1]
        array_structured['node_index'] = index_tuples[:, 0]
        array_structured['face_index'] = index_tuples[:, 1]
        
        return array_structured
    
    def create_Tk_Ok_vector(self, faces_adj_by_edges, tk_points, faces_centroids, bool_boundary_edges, edges):
        
        centroids_of_faces_adj = faces_centroids[faces_adj_by_edges]
        Tk_Ok = centroids_of_faces_adj - tk_points.reshape((tk_points.shape[0], 1, tk_points.shape[1]))
        
        dtype_struc = [('edge_index', np.uint64), ('face_index', np.uint64)]
        bool_internal_edges = ~bool_boundary_edges
        
        n_boundary_edges = bool_boundary_edges.sum()
        n_internal_edges = bool_internal_edges.sum()
        n_lines = n_internal_edges*2 + n_boundary_edges
        
        Tk_Ok_vector = np.zeros((n_lines, 3))
        Tk_Ok_index = np.zeros(n_lines, dtype=dtype_struc)
        
        Tk_Ok_index['edge_index'][0:n_boundary_edges] = edges[bool_boundary_edges]
        Tk_Ok_index['face_index'][0:n_boundary_edges] = faces_adj_by_edges[bool_boundary_edges, 0]
        Tk_Ok_vector[0:n_boundary_edges] = Tk_Ok[bool_boundary_edges, 0]
        
        Tk_Ok_index['edge_index'][n_boundary_edges: n_boundary_edges + n_internal_edges] = edges[bool_internal_edges]
        Tk_Ok_index['face_index'][n_boundary_edges: n_boundary_edges + n_internal_edges] = faces_adj_by_edges[bool_internal_edges, 0]
        Tk_Ok_vector[n_boundary_edges: n_boundary_edges + n_internal_edges] = Tk_Ok[bool_internal_edges, 0]
        
        Tk_Ok_index['edge_index'][n_boundary_edges + n_internal_edges:] = edges[bool_internal_edges]
        Tk_Ok_index['face_index'][n_boundary_edges + n_internal_edges:] = faces_adj_by_edges[bool_internal_edges, 1]
        Tk_Ok_vector[n_boundary_edges+n_internal_edges:] = Tk_Ok[bool_internal_edges, 1]
        
        return Tk_Ok_vector, Tk_Ok_index

    def create_kn_and_kt_Tk_Ok(self, Tk_Ok, Tk_Ok_index, permeability):
        
        R_matrix = np.array([[0, 1],
                             [-1, 0]])
        
        normal_tk_ok = np.matmul(Tk_Ok[:,0:2], R_matrix)
        norm_normal_tk_ok = np.linalg.norm(normal_tk_ok, axis=1)
        
        perm_faces = permeability[Tk_Ok_index['face_index']]
        
        d1 = np.arange(normal_tk_ok.shape[0])
        
        intermediate = np.tensordot(normal_tk_ok, perm_faces, axes=((1), (1)))[d1,d1,:]

        kn = (np.tensordot(intermediate, normal_tk_ok, axes=((1), (1)))[d1,d1])/(norm_normal_tk_ok)
        
        kt = (np.tensordot(intermediate, Tk_Ok[:,0:2], axes=((1), (1)))[d1,d1])/(norm_normal_tk_ok)
        
        dtype_struc = [('edge_index', np.uint64), ('face_index', np.uint64), ('kn_tk_ok', np.float64), ('kt_tk_ok', np.float64)]
        
        resp = np.zeros(len(kn), dtype=dtype_struc)
        resp['edge_index'] = Tk_Ok_index['edge_index']
        resp['face_index'] = Tk_Ok_index['face_index']
        resp['kn_tk_ok'] = kn
        resp['kt_tk_ok'] = kt
         
        return resp
    
    def create_q0_tk_vector(self, edges_adj_by_nodes, tk_points, nodes_centroids):
        
        node_index = []
        edge_index = []
        q0_tk_vector = []
        
        for i, edges_adj in enumerate(edges_adj_by_nodes):
            tk_edges = tk_points[edges_adj]
            node_centroid = nodes_centroids[i]
            q0_tk = tk_edges - node_centroid
            q0_tk_vector.append(q0_tk)
            node_index.append(np.repeat(i, len(edges_adj)))
            edge_index.append(edges_adj)
        
        node_index = np.concatenate(node_index)
        edge_index = np.concatenate(edge_index)
        q0_tk_vector = np.concatenate(q0_tk_vector)
        
        dtype_struc = [('node_index', np.uint64), ('edge_index', np.uint64)]
        q0_tk_index = np.zeros(len(node_index), dtype=dtype_struc)
        q0_tk_index['edge_index'] = edge_index
        q0_tk_index['node_index'] = node_index
        
        return q0_tk_vector, q0_tk_index

    def create_neta_kn_and_kt_Q0_Tk(self, q0_tk_vector, q0_tk_index, faces_adj_by_edges, permeability, bool_boundary_edges, edges, h_distance):
        
        R_matrix = np.array([[0, 1],
                             [-1, 0]])
        
        normal_q0_tk = np.matmul(q0_tk_vector[:,0:2], R_matrix)
        norm_normal_q0_tk = np.linalg.norm(normal_q0_tk, axis=1)
        
        
        bool_internal_edges = ~bool_boundary_edges
        internal_edges = edges[bool_internal_edges]
        
        test1 = np.isin(q0_tk_index['edge_index'], internal_edges)
        
        q0_tk_internal_index = q0_tk_index[test1]
        q0_tk_internal_vector = q0_tk_vector[test1]
        normal_q0_tk_internal = normal_q0_tk[test1]
        norm_normal_q0_tk_internal = norm_normal_q0_tk[test1]
        
        node_index = []
        edge_index = []
        face_index = []
        kn_resp = []
        kt_resp = []
        neta = []
        
        for node, edge, q0_tk, normal_q0_tk_in, norm_normal_q0_tk_in in zip(q0_tk_internal_index['node_index'], q0_tk_internal_index['edge_index'], q0_tk_internal_vector, normal_q0_tk_internal, norm_normal_q0_tk_internal):
            faces_adj_edge = faces_adj_by_edges[edge]
            node_index.append(np.repeat(node, len(faces_adj_edge)))
            edge_index.append(np.repeat(edge, len(faces_adj_edge)))
            h_dist = h_distance[edge]
            for i, face in enumerate(faces_adj_edge):
                face_index.append(face)
                hd = h_dist[i]             
                perm_face = permeability[face]
                intermediate = np.dot(normal_q0_tk_in, perm_face)
                kn = np.dot(intermediate, normal_q0_tk_in)/(norm_normal_q0_tk_in)
                kt = np.dot(intermediate, q0_tk[0:2])/(norm_normal_q0_tk_in)
                kn_resp.append(kn)
                kt_resp.append(kt)
                neta.append(norm_normal_q0_tk_in/hd)
        
        test2 = ~test1
        q0_tk_internal_index = q0_tk_index[test2]
        q0_tk_internal_vector = q0_tk_vector[test2]
        normal_q0_tk_internal = normal_q0_tk[test2]
        norm_normal_q0_tk_internal = norm_normal_q0_tk[test2]
        
        for node, edge, q0_tk, normal_q0_tk_in, norm_normal_q0_tk_in in zip(q0_tk_internal_index['node_index'], q0_tk_internal_index['edge_index'], q0_tk_internal_vector, normal_q0_tk_internal, norm_normal_q0_tk_internal):
            face = faces_adj_by_edges[edge][0]
            node_index.append(np.array([node]))
            edge_index.append(np.array([edge]))
            face_index.append(face)
            hd = h_distance[edge][0]             
            perm_face = permeability[face]
            intermediate = np.dot(normal_q0_tk_in, perm_face)
            kn = np.dot(intermediate, normal_q0_tk_in)/(norm_normal_q0_tk_in)
            kt = np.dot(intermediate, q0_tk[0:2])/(norm_normal_q0_tk_in)
            kn_resp.append(kn)
            kt_resp.append(kt)
            neta.append(norm_normal_q0_tk_in/hd)
        
        node_index = np.concatenate(node_index)
        edge_index = np.concatenate(edge_index)
        face_index = np.array(face_index)
        kn_resp = np.array(kn_resp)
        kt_resp = np.array(kt_resp)
        neta = np.array(neta)
        
        dtype_struc = [('node_index', np.uint64), ('edge_index', np.uint64), ('face_index', np.uint64), ('kn_q0_tk', np.float64), ('kt_q0_tk', np.float64), ('neta', np.float64)]
        
        kn_kt_resp = np.zeros(len(kn_resp), dtype=dtype_struc)
        kn_kt_resp['edge_index'] = edge_index
        kn_kt_resp['face_index'] = face_index
        kn_kt_resp['node_index'] = node_index
        kn_kt_resp['kn_q0_tk'] = kn_resp
        kn_kt_resp['kt_q0_tk'] = kt_resp
        kn_kt_resp['neta'] = neta
        
        return kn_kt_resp
        
        
            
        