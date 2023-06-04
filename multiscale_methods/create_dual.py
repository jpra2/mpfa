import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy import sparse as sp

def get_local_delta(nodes_of_edges, nodes_centroids):
    nodes_edges_centroids = nodes_centroids[nodes_of_edges]
    deltas = np.linalg.norm(nodes_edges_centroids[:, 1] - nodes_edges_centroids[:, 0], axis=1)
    return deltas.min()


def create_dual_2d(
        level, 
        centroids, 
        adjacencies, 
        coarse_gids, 
        bool_boundary_edges, 
        nodes_of_edges, 
        nodes_centroids,
        fine_edges,
        faces_adj_by_nodes,
        coarse_adjacencies,
        fine_edges_centroids,
        fine_faces,
        fine_h_dist
    ):
    """Create dual structure

    Args:
        level (_type_): level of dual
        centroids (_type_): finescale x,y centroids of faces
        adjacencies (_type_): finescale adjacencies by edges
        coarse_gids (_type_): coarse gids of finescale
        bool_boundary_edges: bool vector of boundary edges
        nodes_of_edges: vertices of edges
        nodes_centroids: centroids of finescale vertices

        dual_structure: 3 = vertice, 2=edge, 1=face

    """ 
    edges = fine_edges
    local_delta = get_local_delta(nodes_of_edges, nodes_centroids)
    gids = fine_faces
    dual_gids = gids.copy()
    dual_gids[:] = -1

    to_resp = {}
    dual_gid_name = 'dual_id_' + str(level)
    
    dists = np.zeros(adjacencies.shape)
    dists[:, 0] = np.linalg.norm(centroids[adjacencies[:, 0]] - fine_edges_centroids, axis=1)
    dists[:, 1] = np.linalg.norm(centroids[adjacencies[:, 1]] - fine_edges_centroids, axis=1)

    def create_vertices_and_boundary_edges():

        boundary_faces = adjacencies[bool_boundary_edges]
        boundary_faces = boundary_faces[boundary_faces != -1]

        x_min, y_min = centroids.min(axis=0)
        x_max, y_max = centroids.max(axis=0)

        def get_corner_vertices_by_x(x, local_gids, centroids_local):
            vertices_gids = []
            for y in [y_min, y_max]:
                point = np.array([x, y])
                dists = np.linalg.norm(centroids_local - point, axis=1)
                vertice = local_gids[dists <= dists.min()][0]
                vertices_gids.append(vertice)
            return np.array(vertices_gids)

        def get_corner_vertices_by_x_min(x_min, gids, local_delta, centroids):
            local_gids = gids[centroids[:,0] <= x_min + 2*local_delta]
            centroids_local = centroids[local_gids]
            return get_corner_vertices_by_x(x_min, local_gids, centroids_local)
        
        def get_corner_vertices_by_x_max(x_max, gids, local_delta, centroids):
            local_gids = gids[centroids[:,0] >= x_max - 2*local_delta]
            centroids_local = centroids[local_gids]
            return get_corner_vertices_by_x(x_max, local_gids, centroids_local)
        
        def gids_corner_vertices():
        
            corner_vertices = []
            corner_vertices.append(get_corner_vertices_by_x_min(x_min, gids, local_delta, centroids))
            corner_vertices.append(get_corner_vertices_by_x_max(x_max, gids, local_delta, centroids))
            corner_vertices = np.concatenate(corner_vertices)
            return corner_vertices
        
        nodes_in_boundary = np.unique(nodes_of_edges[bool_boundary_edges])
        boundary_faces = np.unique(np.concatenate(faces_adj_by_nodes[nodes_in_boundary]))
        dual_gids[boundary_faces] = 2
        dual_gids[gids_corner_vertices()] = 3

        def create_boundary_dual_vertices_ids(coarse_gids, gids, dual_gids, edges, adjacencies, centroids, bool_boundary_edges, coarse_adj, fine_edges_centroids):
            
            b_vertices = []
            cids_in_boundary = coarse_adj[:,0][coarse_adj[:,1] == -1]
            for cid in np.unique(cids_in_boundary):
                local_ids = gids[coarse_gids == cid]
                if np.any(dual_gids[local_ids] == 3):
                    continue
                
                edges_in_boundary = edges[
                    np.isin(adjacencies[:,0], local_ids) &
                    bool_boundary_edges
                ]

                # centroids_edges_in_boundary = nodes_centroids[nodes_of_edges[edges_in_boundary]]
                # cmean = centroids_edges_in_boundary.mean(axis=1).mean(axis=0)
                centroids_edges_in_boundary = fine_edges_centroids[edges_in_boundary]
                cmean = centroids_edges_in_boundary.mean(axis=0)
                
                dists = centroids[adjacencies[edges_in_boundary, 0]] - cmean
                dists = np.linalg.norm(dists, axis=1)
                vertice = adjacencies[edges_in_boundary, 0][dists <= dists.min()][0]
                b_vertices.append(vertice)
            return np.array(b_vertices)

        b_vertices = create_boundary_dual_vertices_ids(coarse_gids, gids, dual_gids, edges, adjacencies, centroids, bool_boundary_edges, coarse_adjacencies, fine_edges_centroids)
        dual_gids[b_vertices] = 3

        def get_internal_vertices(coarse_gids, dual_gids, centroids, gids):
            coarse_ids_with_vertices = np.unique(coarse_gids[dual_gids == 3])
            coarse_ids_todo = np.setdiff1d(np.unique(coarse_gids), coarse_ids_with_vertices).astype(np.uint64)
            
            vertices = []
            for cid in coarse_ids_todo:
                local_gids = gids[coarse_gids == cid]
                local_centroids = centroids[local_gids]
                dists = np.linalg.norm(local_centroids - local_centroids.mean(axis=0), axis=1)
                vertice = local_gids[dists <= dists.min()][0]
                vertices.append(vertice)

            return np.array(vertices)
        
        dual_gids[get_internal_vertices(coarse_gids, dual_gids, centroids, gids)] = 3

    create_vertices_and_boundary_edges()
    
    def create_internal_dual_edges(level):
        
        def get_Path(Pr, i, j):
            path = [j]
            k = j
            while Pr[i, k] != -9999:
                path.append(Pr[i, k])
                k = Pr[i, k]
            return np.array(path[::-1])
        
        def mount_graph(adjacencies, dists, coarse_gid, coarse_adj_fine, fine_vertice, fine_edge):
            
            test = (coarse_adj_fine[:, 0] == coarse_gid) & (coarse_adj_fine[:, 1] == coarse_gid)
            local_adjacencies = adjacencies[test]
            
            faces = np.unique(local_adjacencies)
            n_faces = len(faces)
            local_faces = np.arange(n_faces)
            
            local_map = np.repeat(-1, faces.max() + 1)
            local_map[faces] = local_faces
            
            local_adjacencies_remapped = local_map[local_adjacencies]
            lr = local_adjacencies_remapped
            
            dists_local = dists[test].sum(axis=1)
            
            vertice_r = local_map[fine_vertice][0]
            edge_r = local_map[fine_edge][0]
            
            local_graph = sp.csr_matrix((dists_local, (lr[:, 0], lr[:, 1])), shape=(n_faces, n_faces))
            D, Pr = shortest_path(local_graph, directed=False, method='D', return_predecessors=True)
            path = get_Path(Pr, vertice_r, edge_r)
            path = np.setdiff1d(path, [vertice_r, edge_r])
            path = faces[np.isin(local_faces, path)]
            
            return path
        
        def mount_graph_2(adjacencies, dists, faces_edge_path, fine_vertice, fine_edge):
            
            test1 = np.isin(adjacencies[:, 0], faces_edge_path)
            test2 = np.isin(adjacencies[:, 1], faces_edge_path)
            test = test1 | test2
            local_adjacencies = adjacencies[test]
            
            faces = np.unique(local_adjacencies)
            n_faces = len(faces)
            local_faces = np.arange(n_faces)
            
            local_map = np.repeat(-1, faces.max() + 1)
            local_map[faces] = local_faces
            
            local_adjacencies_remapped = local_map[local_adjacencies]
            lr = local_adjacencies_remapped
            
            dists_local = dists[test].sum(axis=1)
            
            vertice_r = local_map[fine_vertice][0]
            
            try:
                edge_r = local_map[fine_edge][0]
            except IndexError: 
                import pdb; pdb.set_trace()
            
            local_graph = sp.csr_matrix((dists_local, (lr[:, 0], lr[:, 1])), shape=(n_faces, n_faces))
            D, Pr = shortest_path(local_graph, directed=False, method='D', return_predecessors=True)
            path = get_Path(Pr, vertice_r, edge_r)
            path = np.setdiff1d(path, [vertice_r, edge_r])
            path = faces[np.isin(local_faces, path)]
            
            return path
        
        def create_edges_dep0(gr: nx.Graph, coarse_gids, coarse_adj, centroids, adjacencies, edges, coarse_adj_fine, nodes_of_edges, nodes_centroids, dual_gids, gids):
            defined_coarse_adjs = set()
            dual_edges = []
            cids = np.unique(coarse_gids)
            internal_adj_coarse = coarse_adj[coarse_adj[:, 1] != -1]
            coarse_faces_in_boundary = set(coarse_adj[:,0][coarse_adj[:,1] == -1])

            for cid in cids:
                
                test1 = len(set([cid]) & coarse_faces_in_boundary)>0
                adjs = internal_adj_coarse[
                    (internal_adj_coarse[:,0] == cid) |
                    (internal_adj_coarse[:,1] == cid)
                ]
                adjs = adjs[adjs!=cid]
                cid_edges = edges[
                    (coarse_adj_fine[:,0] == cid) |
                    (coarse_adj_fine[:,1] == cid)
                ]

                gid_vertice_cid = gids[(coarse_gids == cid) & (dual_gids == 3)][0]

                for adj_cid in adjs:
                    test2 = len(set([adj_cid]) & coarse_faces_in_boundary)>0
                    test3 = len(set((cid, adj_cid)) & defined_coarse_adjs)>0

                    if test1 and test2:
                        continue
                    
                    if test3:
                        continue

                    gid_vertice_adj_cid = gids[(coarse_gids == adj_cid) & (dual_gids == 3)][0]
                    cid_edges_adj = edges[
                        (coarse_adj_fine[:,0] == adj_cid) |
                        (coarse_adj_fine[:,1] == adj_cid)
                    ]

                    intersect_edges = np.intersect1d(cid_edges, cid_edges_adj)
                    centroids_intersect_edges = nodes_centroids[nodes_of_edges[intersect_edges]].mean(axis=1)
                    cmean = centroids_intersect_edges.mean(axis=0)
                    dists = np.linalg.norm(centroids_intersect_edges - cmean, axis=1)
                    edge_intersect = intersect_edges[dists <= dists.min()][0]
                    fine_faces = adjacencies[edge_intersect]
                    

                    for face in fine_faces:
                        if coarse_gids[face] == cid:
                            vertice_id = gid_vertice_cid
                        elif coarse_gids[face] == adj_cid:
                            vertice_id = gid_vertice_adj_cid
                        else:
                            raise ValueError
                        
                        edges_local = nx.shortest_path(gr,source=vertice_id,target=face, weight='weight')                            
                        dual_edges.append(edges_local)
                    
                    defined_coarse_adjs.add((cid, adj_cid))
                    defined_coarse_adjs.add((adj_cid, cid))

            dual_edges = np.concatenate(dual_edges)
            dual_edges = np.setdiff1d(dual_edges, gids[dual_gids==3])
            return dual_edges 

        def create_edges(coarse_gids, fine_adjacencies, coarse_adjacencies, fine_edges_centroids, fine_edges, dual_gids, fine_faces, dists, fine_faces_centroids, fine_bool_boundary_edges):
            
            defined_coarse_vols_tuples = set()
            fine_bool_internal_edges = ~fine_bool_boundary_edges
            
            dual_edges = []
            cids = np.unique(coarse_gids)
            
            coarse_adj_fine = coarse_gids.astype(np.int64)[fine_adjacencies]
            coarse_adj_fine[fine_adjacencies[:,1] == -1, 1] = -1
            
            coarse_boundary_faces = np.unique(coarse_adj_fine[:,0][coarse_adj_fine[:, 1] == -1])
            
            cids = np.setdiff1d(cids, coarse_boundary_faces)
            bool_coarse_boundary_edges = coarse_adjacencies[:, 1] == -1
            bool_coarse_internal_edges = ~bool_coarse_boundary_edges            

            for coarse_adj in coarse_adjacencies[bool_coarse_internal_edges]:
                if len(np.intersect1d(coarse_adj, coarse_boundary_faces)) > 1:
                    continue
                
                # coarse_adj = coarse_adjacencies[coarse_adjacencies[:, 1] != -1]
                # coarse_adj = coarse_adj[
                #     ((coarse_adj[:, 0] == cid) & (coarse_adj[:, 1] != cid)) |
                #     ((coarse_adj[:, 1] == cid) & (coarse_adj[:, 0] != cid))
                # ]
                
                # coarse_faces_adj = coarse_adj[coarse_adj != cid]
                cid = coarse_adj[0]
                coarse_face = coarse_adj[1]
                
                # for coarse_face in coarse_faces_adj:
                #     test1 = set((cid, coarse_face)) & defined_coarse_vols_tuples
                #     test2 = set((coarse_face, cid)) & defined_coarse_vols_tuples
                #     if len(test1) > 0 | len(test2) > 0:
                #         continue


                fine_edges_intersection = fine_edges[
                    ((coarse_adj_fine[:, 0] == cid) & (coarse_adj_fine[:, 1] == coarse_face)) |
                    ((coarse_adj_fine[:, 1] == cid) & (coarse_adj_fine[:, 0] == coarse_face))
                ]
                
                if len(fine_edges_intersection) < 3:
                    # import pdb; pdb.set_trace()
                    selected_edge = fine_edges_intersection[0]
                else:
                    gcentroid = np.mean(fine_edges_centroids[fine_edges_intersection], axis=0)
                    dists_ = np.linalg.norm(fine_edges_centroids[fine_edges_intersection] - gcentroid, axis=1)
                    selected_edge = fine_edges_intersection[dists_ <= dists_.min()][0]
                
                fine_faces_selected = fine_adjacencies[selected_edge]
                dual_edges.append(fine_faces_selected.flatten())
                
                fine_face_edge_cid = fine_faces_selected[coarse_gids[fine_faces_selected] == cid]
                fine_face_edge_coarse_face = fine_faces_selected[coarse_gids[fine_faces_selected] == coarse_face]
                
                face_vertice_cid = fine_faces[(dual_gids == 3) & (coarse_gids == cid)]
                face_vertice_coarse_face = fine_faces[(dual_gids == 3) & (coarse_gids == coarse_face)]
                
                test1 = ((fine_adjacencies[:, 0] == face_vertice_cid) | (fine_adjacencies[:, 1] == face_vertice_cid)) & fine_bool_internal_edges
                test2 = ((fine_adjacencies[:, 0] == face_vertice_coarse_face) | (fine_adjacencies[:, 1] == face_vertice_coarse_face)) & fine_bool_internal_edges

                faces_conected_to_vertice_cid = fine_adjacencies[test1][fine_adjacencies[test1] != face_vertice_cid]
                faces_conected_to_vertice_coarse_face = fine_adjacencies[test2][fine_adjacencies[test2] != face_vertice_coarse_face]
                
                dists_f1 = fine_faces_centroids[faces_conected_to_vertice_cid] - fine_faces_centroids[face_vertice_coarse_face]
                dists_f2 = fine_faces_centroids[faces_conected_to_vertice_coarse_face] - fine_faces_centroids[face_vertice_cid]
                dists_f1 = np.linalg.norm(dists_f1, axis=1)
                dists_f2 = np.linalg.norm(dists_f2, axis=1)
                
                edge_connected_to_vertice_cid = faces_conected_to_vertice_cid[dists_f1 <= dists_f1.min()]
                edge_connected_to_vertice_coarse_face = faces_conected_to_vertice_coarse_face[dists_f2 <= dists_f2.min()]
                
                dual_edges.append([edge_connected_to_vertice_cid[0], edge_connected_to_vertice_coarse_face[0]])                
                    
                edges_cid = mount_graph(fine_adjacencies, dists, cid, coarse_adj_fine, edge_connected_to_vertice_cid, fine_face_edge_cid)
                edges_coarse_face = mount_graph(fine_adjacencies, dists, coarse_face, coarse_adj_fine, edge_connected_to_vertice_coarse_face, fine_face_edge_coarse_face)
                # edges_cid = mount_graph_2(fine_adjacencies, dists, edges_cid, face_vertice_cid, fine_face_edge_cid)
                # edges_coarse_face = mount_graph_2(fine_adjacencies, dists, edges_coarse_face, face_vertice_coarse_face, fine_face_edge_coarse_face)
                
                dual_edges.append(np.concatenate([edges_cid, edges_coarse_face]).flatten())                    
                defined_coarse_vols_tuples.add((cid, coarse_face))
            
            return np.concatenate(dual_edges)
       
        dual_edges = create_edges(coarse_gids, adjacencies, coarse_adjacencies, fine_edges_centroids, fine_edges, dual_gids, fine_faces, fine_h_dist, centroids, bool_boundary_edges)
        dual_gids[dual_edges] = 2
    
    create_internal_dual_edges(level)

    dual_gids[dual_gids == -1] = 1

    to_resp.update({
        dual_gid_name: dual_gids
    })

    return to_resp