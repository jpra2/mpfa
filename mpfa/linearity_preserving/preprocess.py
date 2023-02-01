import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

class MpfaLinearityPreservingPreprocess2D:
    
    def create_Tk_point(self, centroids_of_edges_nodes, tal=0.5):
        
        assert (tal > 0) and (tal < 1)
        edges_points = centroids_of_edges_nodes
        Tk = edges_points[:,0]*tal +(1-tal)*edges_points[:, 1]
        
        return Tk
    
    def create_phis_and_thetas(self, nodes, edges, faces, nodes_centroids, faces_centroids, nodes_of_edges, tk_points, edges_of_faces, faces_adj_by_nodes, edges_adj_by_nodes, faces_adj_by_edges):
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
        
        node_index = []
        edge_index = []
        face_index = []
        phis = []
        thetas = []
        
        dtype_phis_and_thetas = [('phi', float), ('theta', float), ('node_index', np.uint64), ('edge_index', np.uint64), ('face_index', np.uint64)]
        
        for node in nodes:
            edges_around_node = edges_adj_by_nodes[node]
            node_centroid = nodes_centroids[node]
            
            for edge in edges_around_node:
                
                faces_adj = faces_adj_by_edges[edge][faces_adj_by_edges[edge] != -1]
                tk_edge = tk_points[edge]
                
                for face in faces_adj:
                    face_centroid = faces_centroids[face]
                    
                    q0ok = np.linalg.norm(face_centroid - node_centroid)
                    q0tk = np.linalg.norm(tk_edge - node_centroid)
                    oktk = np.linalg.norm(face_centroid - tk_edge)
                    
                    cos_theta = (oktk**2 - (q0ok**2 + q0tk**2))/(-2*q0ok*q0tk)
                    cos_phi = (q0tk**2 - (q0ok**2 + oktk**2))/(-2*q0ok*oktk)
                    
                    theta = np.arccos(cos_theta)
                    phi = np.arccos(cos_phi)
                    
                    node_index.append(node)
                    edge_index.append(edge)
                    face_index.append(face)
                    thetas.append(theta)
                    phis.append(phi)
        
        phis = np.array(phis)
        thetas = np.array(thetas)
        node_index = np.array(node_index)
        edge_index = np.array(edge_index)
        face_index = np.array(face_index)
        
        array_structured = np.zeros(len(phis), dtype=dtype_phis_and_thetas)
        
        array_structured['phi'] = phis
        array_structured['theta'] = thetas
        array_structured['node_index'] = node_index
        array_structured['face_index'] = face_index
        array_structured['edge_index'] = edge_index
        
        return array_structured
    
    def create_Tk_Ok_vector(self, faces_adj_by_edges, tk_points, faces_centroids, bool_boundary_edges, edges):
        
        centroids_of_faces_adj = faces_centroids[faces_adj_by_edges]
        Tk_Ok = centroids_of_faces_adj - tk_points.reshape((tk_points.shape[0], 1, tk_points.shape[1]))
        
        dtype_struc = [('edge_index', np.uint64), ('face_index', np.uint64), ('tk_ok_vector', np.float64, (3,))]
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
        
        Tk_Ok_index['tk_ok_vector'][:] = Tk_Ok_vector
        
        return Tk_Ok_index

    def create_kn_and_kt_Tk_Ok(self, Tk_Ok, Tk_Ok_index, permeability):
        
        R_matrix = np.array([[0, 1],
                             [-1, 0]])
        
        normal_tk_ok = np.matmul(Tk_Ok[:,0:2], R_matrix)
        norm_normal_tk_ok = np.linalg.norm(normal_tk_ok, axis=1)
        
        perm_faces = permeability[Tk_Ok_index['face_index']]
        
        d1 = np.arange(normal_tk_ok.shape[0])
        
        intermediate = np.tensordot(normal_tk_ok, perm_faces, axes=((1), (1)))[d1,d1,:]

        kn = (np.tensordot(intermediate, normal_tk_ok, axes=((1), (1)))[d1,d1])/(np.power(norm_normal_tk_ok, 2))
        
        kt = (np.tensordot(intermediate, Tk_Ok[:,0:2], axes=((1), (1)))[d1,d1])/(np.power(norm_normal_tk_ok, 2))

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
        
        dtype_struc = [('node_index', np.uint64), ('edge_index', np.uint64), ('q0_tk_vector', np.float64, (3,))]
        q0_tk_index = np.zeros(len(node_index), dtype=dtype_struc)
        q0_tk_index['edge_index'] = edge_index
        q0_tk_index['node_index'] = node_index
        q0_tk_index['q0_tk_vector'] = q0_tk_vector
        
        return q0_tk_index

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
                kn = np.dot(intermediate, normal_q0_tk_in)/(np.power(norm_normal_q0_tk_in, 2))
                kt = np.dot(intermediate, q0_tk[0:2])/(np.power(norm_normal_q0_tk_in, 2))
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
            kn = np.dot(intermediate, normal_q0_tk_in)/(np.power(norm_normal_q0_tk_in, 2))
            kt = np.dot(intermediate, q0_tk[0:2])/(np.power(norm_normal_q0_tk_in, 2))
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
        
    def create_weights(self, kn_kt_tk_ok, kn_kt_q0_tk, phis_and_thethas, edges_of_nodes, faces_adj_by_nodes, bool_boundary_nodes, nodes, faces_adj_by_edges, edges, bool_boundary_edges, edges_of_faces):
        
        bool_internal_nodes = ~bool_boundary_nodes
        internal_nodes = nodes[bool_internal_nodes]
        
        K_barra_alpha = kn_kt_tk_ok
        K_alpha = kn_kt_q0_tk
        
        node_index = []
        face_index = []
        lambda_value = []
        weights = []
        
        dtype_struc = [('node_index', np.uint64), ('face_index', np.uint64), ('lambda_value', np.float64), ('weight', np.float64)]
        
        for node in nodes:
            faces_node = faces_adj_by_nodes[node]
            edges_node = edges_of_nodes[node]
            n_edges_node = len(edges_node)
            lambdas_node = np.zeros(len(faces_node))
            
            for i, face in enumerate(faces_node):
                
                edges_face_node = self.test_edges_order(
                    np.intersect1d(edges_of_faces[face], edges_node),
                    edges_node,
                    n_edges_node
                )
                
                faces_adj_of_face_by_edges = faces_adj_by_edges[edges_face_node][faces_adj_by_edges[edges_face_node] != face]

                lambda_face = self.define_lambda(K_barra_alpha, K_alpha, node, face, edges_face_node, faces_adj_of_face_by_edges, phis_and_thethas)
                    
                node_index.append(node)
                face_index.append(face)
                lambdas_node[i] = lambda_face
            
            lambda_value.append(lambdas_node)
            weights.append(lambdas_node/lambdas_node.sum())
            
        
        node_index = np.array(node_index)
        face_index = np.array(face_index)
        lambda_value = np.concatenate(lambda_value)
        weights = np.concatenate(weights)
        
        lambdas = np.zeros(len(node_index), dtype=dtype_struc)
        
        lambdas['node_index'] = node_index
        lambdas['face_index'] = face_index
        lambdas['lambda_value'] = lambda_value
        lambdas['weight'] = weights
        
        return lambdas
    
    def define_lambda(self, K_barra_alpha, K_alpha, node, face, edges_face_node, faces_adj_of_face_by_edges, phis_and_thethas):
        
        #######################################
        ### node_face_edge0
        
        K_barra_n_face_edge0, K_barra_t_face_edge0  = self.get_knkt_by_edge_and_face_index(K_barra_alpha, edges_face_node[0], face)
        
        Kn_node_face_edge0, Kt_node_face_edge0, neta_node_face_edge0 = self.get_neta_kn_kt_by_node_face_and_edge_index(K_alpha, node, edges_face_node[0], face)
        
        phi_node_face_edge0, theta_node_face_edge0 = self.get_phi_and_theta(phis_and_thethas, node, edges_face_node[0], face)
        
        ###################
        ### node_face_edge1
        K_barra_n_face_edge1, K_barra_t_face_edge1  = self.get_knkt_by_edge_and_face_index(K_barra_alpha, edges_face_node[1], face)
        
        Kn_node_face_edge1, Kt_node_face_edge1, neta_node_face_edge1 = self.get_neta_kn_kt_by_node_face_and_edge_index(K_alpha, node, edges_face_node[1], face)
        
        phi_node_face_edge1, theta_node_face_edge1 = self.get_phi_and_theta(phis_and_thethas, node, edges_face_node[1], face)
        
        ########################################
        
        ########################################
        ### node_face_adj0_edge0
        
        K_barra_n_faceadj0_edge0, K_barra_t_faceadj0_edge0  = self.get_knkt_by_edge_and_face_index(K_barra_alpha, edges_face_node[0], face)
                    
        Kn_node_faceadj0_edge0, Kt_node_faceadj0_edge0, neta_node_faceadj0_edge0 = self.get_neta_kn_kt_by_node_face_and_edge_index(K_alpha, node, edges_face_node[0], faces_adj_of_face_by_edges[0])
        
        phi_node_faceadj0_edge0, theta_node_faceadj0_edge0 = self.get_phi_and_theta(phis_and_thethas, node, edges_face_node[0], faces_adj_of_face_by_edges[0])
        
        ########################################
        ## node_face_adj1_edge1
        
        K_barra_n_faceadj1_edge1, K_barra_t_faceadj1_edge1  = self.get_knkt_by_edge_and_face_index(K_barra_alpha, edges_face_node[1], faces_adj_of_face_by_edges[1])
            
        Kn_node_faceadj1_edge1, Kt_node_faceadj1_edge1, neta_node_faceadj1_edge1 = self.get_neta_kn_kt_by_node_face_and_edge_index(K_alpha, node, edges_face_node[1], faces_adj_of_face_by_edges[1])
        
        phi_node_faceadj1_edge1, theta_node_faceadj1_edge1 = self.get_phi_and_theta(phis_and_thethas, node, edges_face_node[1], faces_adj_of_face_by_edges[1])
        
        ################################
        
        phi_k = self.define_phik(
            K_barra_n_faceadj0_edge0,
            phi_node_faceadj0_edge0,
            K_barra_n_face_edge0,
            phi_node_face_edge0,
            K_barra_t_faceadj0_edge0,
            K_barra_t_face_edge0,
            Kn_node_faceadj0_edge0,
            theta_node_faceadj0_edge0,
            Kn_node_face_edge0,
            theta_node_face_edge0,
            Kt_node_faceadj0_edge0,
            Kt_node_face_edge0           
        )
        
        phi_k_plus1 = self.define_phik(
            K_barra_n_face_edge1,
            phi_node_face_edge1,
            K_barra_n_faceadj1_edge1,
            phi_node_faceadj1_edge1,
            K_barra_t_face_edge1,
            K_barra_t_faceadj1_edge1,
            Kn_node_face_edge1,
            theta_node_face_edge1,
            Kn_node_faceadj1_edge1,
            theta_node_faceadj1_edge1,
            Kt_node_face_edge1,
            Kt_node_faceadj1_edge1
        )
        
        term1 = Kn_node_face_edge0*neta_node_face_edge0*phi_k
        term2 = Kn_node_face_edge1*neta_node_face_edge1*phi_k_plus1
        term3 = K_barra_n_face_edge0*self.get_cotangent_angle(phi_node_face_edge0 + theta_node_face_edge0) + K_barra_n_face_edge1*self.get_cotangent_angle(phi_node_face_edge1 + theta_node_face_edge1)
        term4 = K_barra_t_face_edge0 - K_barra_t_face_edge1
        
        lambda_k = term1 + term2 + term3 + term4
        
        return lambda_k
    
    def define_phik(self, K_barra_n2_facekminus1, phi2_facekminus1, K_barra_n1_facek, phi1_facek, K_barra_t2_facekminus1, K_barra_t1_facek, K_n2_facekminus1, theta2_facekminus1, K_n1_facek, theta1_facek, K_t2_facekminus1, K_t1_facek):
        
        cot_phi2_facekminus1 = self.get_cotangent_angle(phi2_facekminus1)
        cot_phi1_facek = self.get_cotangent_angle(phi1_facek)
        cot_theta2_facekminus1 = self.get_cotangent_angle(theta2_facekminus1)
        cot_theta1_facek = self.get_cotangent_angle(theta1_facek)
        
        numer = K_barra_n2_facekminus1*cot_phi2_facekminus1 + K_barra_n1_facek*cot_phi1_facek + K_barra_t2_facekminus1 - K_barra_t1_facek
        
        den = K_n2_facekminus1*cot_theta2_facekminus1 + K_n1_facek*cot_theta1_facek - K_t2_facekminus1 + K_t1_facek
        
        phi_k = numer/den
        
        return phi_k
        
    def get_knkt_by_edge_and_face_index(self, kn_kt, edge_index, face_index):
        
        if self.test_face_in_boundary(face_index):
            return 0, 0
        
        test = (kn_kt['edge_index'] == edge_index) & (kn_kt['face_index'] == face_index)
        kn, kt = kn_kt[['kn_tk_ok', 'kt_tk_ok']][test][0]
        
        return kn, kt
    
    def get_neta_kn_kt_by_node_face_and_edge_index(self, kn_kt, node_index, edge_index, face_index):
        
        if self.test_face_in_boundary(face_index):
            return 0, 0, 0
        
        test = (kn_kt['node_index'] == node_index) & (kn_kt['face_index'] == face_index) & (kn_kt['edge_index'] == edge_index)
        
        kn, kt, neta = kn_kt[['kn_q0_tk', 'kt_q0_tk', 'neta']][test][0]
        
        return kn, kt, neta
        
    def get_phi_and_theta(self, phis_and_thetas, node_index, edge_index, face_index):
        
        if self.test_face_in_boundary(face_index):
            return 0, 0
        
        test = (phis_and_thetas['node_index'] == node_index) & (phis_and_thetas['face_index'] == face_index) & (phis_and_thetas['edge_index'] == edge_index)
        
        phi, theta = phis_and_thetas[['phi', 'theta']][test][0]
        return phi, theta
    
    def test_edges_order(self, edges_face_node, edges_node, n_edges_node):
        posedge0 = np.argwhere(edges_node == edges_face_node[0])[0][0]
        posedge1 = np.argwhere(edges_node == edges_face_node[1])[0][0]
        
        if posedge0 == n_edges_node-1 and posedge1 == 0:
            pass
        elif posedge1 == n_edges_node-1 and posedge0 == 0:
            edges_face_node[:] = np.flip(edges_face_node)
        elif posedge0 > posedge1:
            edges_face_node[:] = np.flip(edges_face_node)
        
        return edges_face_node
    
    def test_face_in_boundary(self, face):
        if face == -1:
            return True
    
    def get_cotangent_angle(self, angle):
        if angle == 0:
            return 0
        else:
            return 1/np.tan(angle)
        
            