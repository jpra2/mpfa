from datamanager.arraydatamanager import SuperArrayManager

class Weight(SuperArrayManager):
    pass

class CalculateGlsWeight2D:

    def __init__(self, **kwargs):
        """ Initialize the class

        @param kwargs: dict with the keys:
            adjacencies: faces adjacencies,
            faces: global ids of faces,
            nodes_centroids: centroids of mesh nodes,
            edges: global ids of mesh edges,
            bool_boundary_edges: bool vector of len(edges) with true in boundary edges,
        """
        self.adjacencies = kwargs.get('adjacencies')
        self.faces = kwargs.get('faces')
        self.faces = kwargs.get('faces')
        self.nodes_centroids = kwargs.get('nodes_centroids')
        self.edges = kwargs.get('edges')
        self.bool_boundary_edges = kwargs.get('bool_boundary_edges')



