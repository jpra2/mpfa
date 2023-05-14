import numpy as np
import networkx as nx

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
        edges,
        faces_adj_by_nodes,
        nodes_of_faces,
        coarse_adjacencies
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
    local_delta = get_local_delta(nodes_of_edges, nodes_centroids)
    gids = np.arange(len(centroids))
    min_delta = local_delta/10
    dual_gids = gids.copy()
    dual_gids[:] = -1

    to_resp = {}
    coarse_adj_name = 'coarse_adj_' + str(level)
    coarse_adj_fine_name = 'coarse_adj_fine_' + str(level)
    dual_gid_name = 'dual_id_' + str(level)

    def create_vertices_and_boundary_edges():

        resp = {}

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

        def define_fine_edges_in_coarse_volumes_adjacencies(level, gids, coarse_gids, adjacencies, edges, bool_boundary_edges):

            coarse_adj_fine_name = 'coarse_adj_fine'
            coarse_adj_name = 'coarse_adj'

            bool_internal_edges = ~bool_boundary_edges
            coarse_adj_fine = adjacencies.copy()
            coarse_adj_fine[bool_internal_edges] = coarse_gids[adjacencies[bool_internal_edges]]
            coarse_adj_fine[bool_boundary_edges,0] = coarse_gids[adjacencies[bool_boundary_edges, 0]]

            coarse_adj = set()

            for cid in np.unique(coarse_gids):
                local_adj = coarse_adj_fine[
                    ((coarse_adj_fine[:,0] == cid) | (coarse_adj_fine[:,1] == cid)) & 
                    (coarse_adj_fine[:,0] != coarse_adj_fine[:,1])
                ]
                local_adj = np.unique(local_adj, axis=0)

                for adj in local_adj:
                    tuple_adj = tuple(adj)
                    test1 = {tuple_adj} & coarse_adj
                    test2 = {tuple(reversed(tuple_adj))} & coarse_adj
                    # import pdb; pdb.set_trace()    
                    if (len(test1) > 0) | (len(test2) > 0):
                        continue
                    else:
                        coarse_adj.add(tuple_adj)

            coarse_adj = np.array([np.array(i) for i in coarse_adj])
            
            return {
                coarse_adj_name: coarse_adj,
                coarse_adj_fine_name: coarse_adj_fine
            }

        def create_boundary_dual_vertices_ids(coarse_gids, gids, dual_gids, edges, adjacencies, nodes_of_edges, nodes_centroids, centroids, bool_boundary_edges, coarse_adj):
            
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

                centroids_edges_in_boundary = nodes_centroids[nodes_of_edges[edges_in_boundary]]
                cmean = centroids_edges_in_boundary.mean(axis=1).mean(axis=0)
                dists = centroids[adjacencies[edges_in_boundary, 0]] - cmean
                dists = np.linalg.norm(dists, axis=1)
                vertice = adjacencies[edges_in_boundary, 0][dists <= dists.min()][0]
                b_vertices.append(vertice)
            return np.array(b_vertices)

        resp.update(define_fine_edges_in_coarse_volumes_adjacencies(level, gids, coarse_gids, adjacencies, edges, bool_boundary_edges))
        b_vertices = create_boundary_dual_vertices_ids(coarse_gids, gids, dual_gids, edges, adjacencies, nodes_of_edges, nodes_centroids, centroids, bool_boundary_edges, resp['coarse_adj'])
        dual_gids[b_vertices] = 3

        def get_internal_vertices(coarse_gids, dual_gids, centroids, gids):
            coarse_ids_with_vertices = np.unique(coarse_gids[dual_gids == 3])
            coarse_ids_todo = np.setdiff1d(np.unique(coarse_gids), coarse_ids_with_vertices)
            
            vertices = []
            for cid in coarse_ids_todo:
                local_gids = gids[coarse_gids == cid]
                local_centroids = centroids[local_gids]
                dists = np.linalg.norm(local_centroids - local_centroids.mean(axis=0), axis=1)
                vertice = local_gids[dists <= dists.min()][0]
                vertices.append(vertice)

            return np.array(vertices)
        
        dual_gids[get_internal_vertices(coarse_gids, dual_gids, centroids, gids)] = 3
        
        return {
            coarse_adj_name: resp['coarse_adj'],
            coarse_adj_fine_name: resp['coarse_adj_fine']
        }

    to_resp.update(create_vertices_and_boundary_edges())
    

    def create_internal_dual_edges(level):
        

        def mount_graph(adjacencies, bool_boundary_edges, centroids, nodes_of_faces, nodes_centroids):
            gr = nx.Graph()
            bool_internal_edges = ~bool_boundary_edges
            adj_internal = adjacencies[bool_internal_edges]

            for adj in adj_internal:
                dist = np.linalg.norm(centroids[adj[1]] - centroids[adj[0]])
                # cmin0 = nodes_centroids[nodes_of_faces[adj[0]]].min(axis=0)
                # cmax0 = nodes_centroids[nodes_of_faces[adj[0]]].max(axis=0)
                # cmin1 = nodes_centroids[nodes_of_faces[adj[1]]].min(axis=0)
                # cmax1 = nodes_centroids[nodes_of_faces[adj[1]]].max(axis=0)
                # points = np.array([cmin0, cmax0, cmin1, cmax1])
                # dists = []
                # for point in points:
                #     dists.append(np.linalg.norm(points - point, axis=1))
                # dist = np.concatenate(dists).max()
                gr.add_edge(adj[0], adj[1], weight=dist)
            
            return gr
        
        def create_edges(gr: nx.Graph, coarse_gids, coarse_adj, centroids, adjacencies, edges, coarse_adj_fine, nodes_of_edges, nodes_centroids, dual_gids, gids):
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

        
        gr = mount_graph(adjacencies, bool_boundary_edges, centroids, nodes_of_faces, nodes_centroids)
        dual_edges = create_edges(gr, coarse_gids, to_resp[coarse_adj_name], centroids, adjacencies, edges, to_resp[coarse_adj_fine_name], nodes_of_edges, nodes_centroids, dual_gids, gids)
        dual_gids[dual_edges] = 2
    
    create_internal_dual_edges(level)

    dual_gids[dual_gids == -1] = 1

    to_resp.update({
        dual_gid_name: dual_gids
    })

    return to_resp