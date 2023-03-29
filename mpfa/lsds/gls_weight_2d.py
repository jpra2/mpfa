from datamanager.arraydatamanager import SuperArrayManager
import numpy as np

class Weight(SuperArrayManager):
    pass

class CalculateGlsWeight2D:

    def __init__(self, **kwargs):
        """ Initialize the class

        @param kwargs: dict with the keys:
            adjacencies: faces adjacencies of edges,
            faces: global ids of faces,
            nodes_of_nodes: nodes adjacencies of nodes
            edges_of_nodes: edges adjacencies of nodes
            nodes_of_edges: nodes adjacencies of edges
            faces_of_nodes: faces adjacencies of nodes
            nodes_centroids: centroids of mesh nodes,
            edges: global ids of mesh edges,
            bool_boundary_edges: bool vector of len(edges) with true in boundary edges,
            bool_boundary_nodes: bool vector of len(nodes) with true in boundary nodes
            nodes: global id of mesh nodes
        """

        self.adjacencies = kwargs.get('adjacencies')
        self.faces = kwargs.get('faces')
        self.nodes_of_nodes = kwargs.get('nodes_of_nodes')
        self.edges_of_nodes = kwargs.get('edges_of_nodes')
        self.nodes_of_edges = kwargs.get('nodes_of_edges')
        self.faces_of_nodes = kwargs.get('faces_of_nodes')
        self.nodes_centroids = kwargs.get('nodes_centroids')
        self.edges = kwargs.get('edges')
        self.bool_boundary_edges = kwargs.get('bool_boundary_edges')
        self.bool_boundary_nodes = kwargs.get('bool_boundary_nodes')
        self.bool_internal_nodes = ~self.bool_boundary_nodes
        self.nodes = kwargs.get('nodes')

    def get_weight(self):

        for node in self.nodes[self.bool_internal_nodes]:
            nodes_adj = self.nodes_of_nodes[node]
            n_m1 = len(nodes_adj)
            m1_local = np.zeros((n_m1, 2*n_m1))            
            edges_adj = self.edges_of_nodes[node]
            centroid_node = self.nodes_centroids[node]
            centroids_nodes_adj = self.nodes_centroids[nodes_adj]
            faces_adj = self.faces_of_nodes[node]
            v_dists_nodes = centroids_nodes_adj - centroid_node
            for i in range(n_m1):
                m1_local[i,2*i:2*i+2] = v_dists_nodes[i]
            
            import pdb; pdb.set_trace()








