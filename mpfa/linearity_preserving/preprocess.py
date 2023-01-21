
class MpfaLinearityPreservingPreprocess2D:
    
    def create_Tk_point(self, centroids_of_edges_nodes, tal=0.5):
        
        assert (tal > 0) and (tal < 1)
        edges_points = centroids_of_edges_nodes
        Tk = edges_points[:,0]*tal +(1-tal)*edges_points[:, 1]
        return Tk
    
    def run(self, centroids_of_edges_nodes):
        Tk = self.create_Tk_point(centroids_of_edges_nodes)
        
        return {
            'Tk_point': Tk
        }