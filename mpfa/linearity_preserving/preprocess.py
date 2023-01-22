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