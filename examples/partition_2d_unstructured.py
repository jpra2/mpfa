import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from datamanager.mesh_data import MeshData
from definitions import defpaths
import numpy as np
import pandas as pd
from multiscale_methods.create_dual import create_dual_2d
from sklearn.cluster import KMeans

###
## run the code examples.create_cube_unstructured for generate mesh
###

mesh_path = os.path.join(defpaths.mesh, '2d_unstructured.msh')
mesh_name = 'partition_square_unstructured_test'

def create_initial_mesh_properties(mesh_path, mesh_name):
    
    mesh_create = CreateMeshProperties()
    mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
    mesh_properties: MeshProperty = mesh_create.create_2d_mesh_data()
    return mesh_properties

def load_mesh_properties(mesh_name):
    
    mesh_properties = MeshProperty()
    mesh_properties.insert_mesh_name([mesh_name])
    mesh_properties.load_data()
    return mesh_properties

def create_partition_dep0():
    A = nx.Graph()
    all_adjs = mesh_properties.faces_adj_by_edges
    adjs = all_adjs[(all_adjs[:,0] != -1) & (all_adjs[:,1] != -1)]
    A.add_edges_from(adjs)
    # G = metis.networkx_to_metis(A)
    G = A
    G.graph['edge_weight_attr']='weight' 

    (edgecuts, parts) = metis.part_graph(G, nparts=2, recursive=True)
    gids = np.array(parts)
    return gids

def create_partition_kmeans(x_centroids, y_centroids):
    # Loading of data set into 'cluster'.
    # cluster = pd.read_csv('k-means clustering.csv')
    
    kmeans_tol = 9e-5
    k = 1000 # Number of clusters
    number_of_iterations = 1000000

    cluster = pd.DataFrame({
        'x': x_centroids,
        'y': y_centroids
    })

    rows = cluster.shape[0] # 'rows' contains the total number of rows in cluster data.
    cols = cluster.shape[1] # 'cols' contains the total number of columns in cluster data.
    
    centroids = cluster.loc[np.random.randint(1,rows+1,k)] # Randomly initialises 'k' no. of centroids.
    centroids['new'] = list(range(1,k+1)) 
    centroids.set_index('new',inplace = True) # New indices 1 to k are set for the dataframe 'centroids'.
    d = np.random.rand(rows) # Initialization of d which would contain the centroid number closest to data point.

    epsilon = list(range(number_of_iterations)) # 'epsilon' is the sum of squares of distances between points and centroid of a cluster for each iteration
    epsilon_0 = 0
    i = 0
    stop = False

    while (i <= number_of_iterations) and (stop == False): # This 'for' loop is for iterations.

        for j in range(0,rows): # This 'for' loop finds the centroid number closest to the data point.
            d[j] = ((centroids - cluster.loc[j])**2).sum(axis = 1).idxmin()
        cluster['centroid number'] = d # A new column 'centroid number' is added to dataframe 'cluster'.

        # plt.subplots_adjust(bottom=0.1, right=2, top=0.9) # Adjusts the subplot.
        # plt.subplot(1,number_of_iterations,i+1)
        # sns.scatterplot(x = 'x',y = 'y',data = cluster,hue = 'centroid number',legend = 'full') # Scatter plot is plotted with differentiating factor as 'centroid number'
        # plt.legend(bbox_to_anchor=(1, 1), loc=4, borderaxespad=0.5) #Adjusts the legend box.
        
        mean_x = list(range(k)) # Initialisation of 'mean_x' which will store mean of 'x' values of each cluster.
        mean_y = list(range(k)) # Initialisation of 'mean_y' which will store mean of 'y' values of each cluster.
        for m in range(0,k): # This 'for' loop calculates mean of 'x' and 'y' values of each cluster.
            mean_x[m] = cluster[cluster['centroid number'] == (m+1)]['x'].mean()
            mean_y[m] = cluster[cluster['centroid number'] == (m+1)]['y'].mean()
        centroids.replace(list(centroids['x']),mean_x,inplace = True) # The 'centroids' are replaced with the new values.
        centroids.replace(list(centroids['y']),mean_y,inplace = True)
        
        z = list(range(k)) # Initialisation of z  and centroid of each cluster.
        for p in range(0,k): # This 'for' loop calculates square of distances between data points and centroid of each cluster.
            z[p] = ((cluster[cluster['centroid number'] == p+1][['x','y']] - centroids.iloc[p])**2).values.sum()
        epsilon[i] = sum(z) # 'epsilon' is sum of squares of distances between points and centroid of a cluster for each iteration
        test = abs(epsilon_0 - epsilon[i])
        print(epsilon[i])
        if test < kmeans_tol:
            stop = True

        epsilon_0 = epsilon[i]
    
    gids = cluster['centroid number'].to_numpy().astype(int) - 1
    cents = np.zeros((len(centroids['x']), 2))
    cents[:,0] = centroids['x'].to_numpy()
    cents[:,1] = centroids['y'].to_numpy()
    import pdb; pdb.set_trace()
    return gids, cents

def create_partition_kmeans_v2(centroids, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=10000, tol=1e-4).fit(centroids)

    coarse_gids = kmeans.labels_.copy()
    corase_centroids = kmeans.cluster_centers_.copy()

    return coarse_gids, corase_centroids

def create_partition(level, x_centroids, y_centroids, faces_adjacencies):
    gids, coarse_centroids = create_partition_kmeans(x_centroids, y_centroids)    
    return gids, coarse_centroids

def create_coarse_adjacencies(level, fine_adjacencies, gids_level, bool_boundary_edges, edges, fine_edges_centroids):
    
    resp = dict()
    bool_intersect_coarse_edges = np.full(len(edges), False, dtype=bool)
    bool_internal_coarse_edges = bool_intersect_coarse_edges.copy()
    
    bool_internal_edges = ~bool_boundary_edges
    
    adj_test = gids_level[fine_adjacencies]
    test_intersect = adj_test[:,0] != adj_test[:,1]
    
    bool_intersect_coarse_edges[:] = test_intersect & bool_internal_edges
    
    test_intersect = ~test_intersect
    
    bool_internal_coarse_edges[:] = test_intersect & bool_internal_edges
    
    resp.update({
        'bool_intersect_fine_edges_' + str(level): bool_intersect_coarse_edges,
        'bool_internal_fine_edges_' + str(level): bool_internal_coarse_edges
    })
    
    cids = np.unique(gids_level)
    
    coarse_adj_fine = adj_test[bool_intersect_coarse_edges]
    
    coarse_adj = set()
    
    for cid in cids:
        local_adj = coarse_adj_fine[
            ((coarse_adj_fine[:,0] == cid) | (coarse_adj_fine[:,1] == cid))
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
    
    coarse_adj_fine = adj_test[bool_boundary_edges]
    local_adj = np.unique(coarse_adj_fine, axis=0)
    local_adj[:, 1] = -1
    
    coarse_adj = np.vstack([local_adj, coarse_adj])
    resp.update({
        'adjacencies_' + str(level): coarse_adj
    })
    
    bool_boundary_edges_coarse = coarse_adj[:, 1] == -1
    
    resp.update({
        'bool_boundary_edges_' + str(level): bool_boundary_edges_coarse
    })
    
    bool_internal_edges_coarse = ~bool_boundary_edges_coarse
    coarse_edges = np.arange(len(coarse_adj))
    resp.update({
        'edges_' + str(level): coarse_edges
    })
    
    coarse_edges_centroids = np.zeros((len(coarse_edges), 2))
    for edge in coarse_edges[bool_internal_edges_coarse]:
        coarse_faces_adj = coarse_adj[edge]
        test = (
            (adj_test[:, 0] == coarse_faces_adj[0]) & (adj_test[:, 1] == coarse_faces_adj[1]) |
            (adj_test[:, 0] == coarse_faces_adj[1]) & (adj_test[:, 1] == coarse_faces_adj[0])
        )
        
        centroid_coarse_edge = np.mean(fine_edges_centroids[test], axis=0)
        coarse_edges_centroids[edge] = centroid_coarse_edge
    
    for edge in coarse_edges[bool_boundary_edges_coarse]:
        coarse_face_adj = coarse_adj[edge][0]
        test = (
            ((adj_test[:,0] == coarse_face_adj) | (adj_test[:,1] == coarse_face_adj)) & bool_boundary_edges
        )
        
        centroid_coarse_edge = np.mean(fine_edges_centroids[test], axis=0)
        coarse_edges_centroids[edge] = centroid_coarse_edge
    
    resp.update({
        'edges_centroids_' + str(level): coarse_edges_centroids
    })
            
    return resp

def create_coarse_volumes(level, centroids, n_clusters):
    
    gid_name = 'gid_' + str(level)
    coarse_centroids_name = 'centroids_' + str(level)
    adjacencies_coarse_name = 'adjacencies_' + str(level)
    intersect_fine_edges_name = 'intersect_fine_faces_' + str(level)
    internal_fine_edges_name = 'internal_fine_faces_' + str(level)
    coarse_edges_name = 'edges_' + str(level)

    coarse_gids, coarse_centroids = create_partition_kmeans_v2(
        centroids,
        n_clusters
    )

    return {
        gid_name: coarse_gids,
        coarse_centroids_name: coarse_centroids
    }

def create_coarse_nodes(level, coarse_adjacencies, fine_faces_adj_by_nodes, gids_level, fine_nodes, fine_bool_boundary_nodes, fine_nodes_of_edges, fine_bool_boundary_edges, bool_intersect_fine_edges_level, fine_adjacencies, coarse_bool_boundary_edges, fine_edges, coarse_edges):
    
    resp = dict()
    cids = np.unique(gids_level)
    
    coarse_nodes = []
    coarse_faces_adj_by_nodes = []
    node_count = 0
    fine_nodes_in_coarse_intersection = []
    coarse_nodes_of_edges = np.zeros((len(coarse_adjacencies), 2), dtype=np.uint64)
    
    nodes_in_intersect_edges = np.unique(fine_nodes_of_edges[bool_intersect_fine_edges_level])
    nodes_in_intersect_edges_in_boundary = np.intersect1d(fine_nodes[fine_bool_boundary_nodes], nodes_in_intersect_edges).astype(np.uint64)
    
    for node in nodes_in_intersect_edges_in_boundary:
        fine_nodes_in_coarse_intersection.append(node)
        coarse_nodes.append(node_count)
        fine_faces_adj_node = fine_faces_adj_by_nodes[node]
        coarse_faces_adj_by_node = np.unique(gids_level[fine_faces_adj_node]).astype(np.uint64)
        coarse_faces_adj_by_nodes.append(coarse_faces_adj_by_node)        
        node_count += 1
    
    nodes_in_intersect_edges_dif = np.setdiff1d(nodes_in_intersect_edges, nodes_in_intersect_edges_in_boundary)
    
    for node in nodes_in_intersect_edges_dif:
        fine_faces_adj_node = fine_faces_adj_by_nodes[node]
        coarse_faces_adj_by_node = np.unique(gids_level[fine_faces_adj_node]).astype(np.uint64)
        if len(coarse_faces_adj_by_node) < 3:
            continue
        fine_nodes_in_coarse_intersection.append(node)
        coarse_nodes.append(node_count)
        coarse_faces_adj_by_nodes.append(coarse_faces_adj_by_node)        
        node_count += 1
    
    fine_nodes_in_coarse_intersection = np.array(fine_nodes_in_coarse_intersection).astype(np.uint64)
    coarse_faces_adj_by_nodes = np.array(coarse_faces_adj_by_nodes, dtype='O')
    coarse_nodes = np.array(coarse_nodes, dtype=np.uint64)
    
    coarse_adj_test = gids_level[fine_adjacencies]
    
    coarse_bool_internal_edges = ~coarse_bool_boundary_edges
    for coarse_edge in coarse_edges[coarse_bool_internal_edges]:
        coarse_faces_adj = coarse_adjacencies[coarse_edge]
        fine_edges_in_intersection = fine_edges[
            ((coarse_adj_test[:, 0] == coarse_faces_adj[0]) & (coarse_adj_test[:, 1] == coarse_faces_adj[1])) |
            ((coarse_adj_test[:, 0] == coarse_faces_adj[1]) & (coarse_adj_test[:, 1] == coarse_faces_adj[0]))
        ]
        nodes_of_edges_in_intersecion = np.unique(fine_nodes_of_edges[fine_edges_in_intersection])
        fine_extremities_nodes = np.intersect1d(nodes_of_edges_in_intersecion, fine_nodes_in_coarse_intersection)
        coarse_extremities_nodes = coarse_nodes[np.isin(fine_nodes_in_coarse_intersection, fine_extremities_nodes)]
        try:
            coarse_nodes_of_edges[coarse_edge] = coarse_extremities_nodes
        except ValueError:
            other_nodes = []
            ft = fine_faces_adj_by_nodes[fine_extremities_nodes]
            for i, fy in enumerate(ft):
                coarse_ids_fy = gids_level[fy]
                if len(set(coarse_ids_fy) & set(coarse_faces_adj)) == 2:
                    other_nodes.append(fine_extremities_nodes[i])
            other_nodes = np.array(other_nodes)
            coarse_extremities_nodes = coarse_nodes[np.isin(fine_nodes_in_coarse_intersection, other_nodes)]
            coarse_nodes_of_edges[coarse_edge] = coarse_extremities_nodes
    
    for coarse_edge in coarse_edges[coarse_bool_boundary_edges]:
        coarse_face_adj = coarse_adjacencies[coarse_edge][0]
        fine_edges_in_intersection = fine_edges[
            (coarse_adj_test[:, 0] == coarse_face_adj) & fine_bool_boundary_edges
        ]
        nodes_of_edges_in_intersecion = np.unique(fine_nodes_of_edges[fine_edges_in_intersection])
        fine_extremities_nodes = np.intersect1d(nodes_of_edges_in_intersecion, fine_nodes_in_coarse_intersection)
        coarse_extremities_nodes = coarse_nodes[np.isin(fine_nodes_in_coarse_intersection, fine_extremities_nodes)]
        coarse_nodes_of_edges[coarse_edge] = coarse_extremities_nodes
    
    resp = {
        'nodes_' + str(level): coarse_nodes,
        'faces_adj_by_nodes_' + str(level): coarse_faces_adj_by_nodes,
        'nodes_of_edges_' + str(level): coarse_nodes_of_edges
    }
    
    return resp
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
        
            
        
    
    
    
    
    import pdb; pdb.set_trace()
    
    
# mesh_properties = create_initial_mesh_properties(mesh_path, mesh_name)
# mesh_properties.export_data()

# mesh_properties = load_mesh_properties(mesh_name)
# fine_edges_nodes_centroids = mesh_properties.nodes_centroids[
#     mesh_properties.nodes_of_edges[
#         mesh_properties.edges
#     ]
# ]
# edges_centroids = np.mean(fine_edges_nodes_centroids, axis=1)
# mesh_properties.insert_data({
#     'edges_centroids': edges_centroids
# })
# mesh_properties.export_data()

# mesh_properties = load_mesh_properties(mesh_name)
# mesh_properties.insert_data(
#     create_coarse_volumes(
#         1,
#         mesh_properties.faces_centroids[:, 0:2],
#         2000
#     )
# )
# mesh_properties.export_data()

# mesh_properties = load_mesh_properties(mesh_name)
# mesh_properties.insert_data(
#     create_coarse_adjacencies(
#         1, 
#         mesh_properties.faces_adj_by_edges,
#         mesh_properties.gid_1,
#         mesh_properties.bool_boundary_edges,
#         mesh_properties.edges,
#         mesh_properties.edges_centroids[:, 0:2]
#     )
# )
# mesh_properties.export_data()

mesh_properties = load_mesh_properties(mesh_name)


create_coarse_nodes(
    1,
    mesh_properties.adjacencies_1,
    mesh_properties.faces_adj_by_nodes,
    mesh_properties.gid_1,
    mesh_properties.nodes,
    mesh_properties.bool_boundary_nodes,
    mesh_properties.nodes_of_edges,
    mesh_properties.bool_boundary_edges,
    mesh_properties.bool_intersect_fine_edges_1,
    mesh_properties.faces_adj_by_edges,
    mesh_properties.bool_boundary_edges_1,
    mesh_properties.edges,
    mesh_properties.edges_1
)

mesh_properties.insert_data(
    create_dual_2d(
        1,
        mesh_properties.faces_centroids[:, 0:2],
        mesh_properties.faces_adj_by_edges,
        mesh_properties.gid_1,
        mesh_properties.bool_boundary_edges,
        mesh_properties.nodes_of_edges,
        mesh_properties.nodes_centroids[:, 0:2],
        mesh_properties.edges,
        mesh_properties.faces_adj_by_nodes,
        mesh_properties.nodes_of_faces,
        mesh_properties.adjacencies_1
    )
)

mesh_data = MeshData(mesh_path=mesh_path)
mesh_data.create_tag('dual_id', data_type='int')
mesh_data.insert_tag_data('dual_id', mesh_properties.dual_id_1, elements_type='faces', elements_array=mesh_properties.faces)
mesh_data.create_tag('coarse_id', data_type='int')
mesh_data.insert_tag_data('coarse_id', mesh_properties.gid_1, elements_type='faces', elements_array=mesh_properties.faces)
mesh_data.export_all_elements_type_to_vtk(export_name='test_gids', element_type='faces')

import pdb; pdb.set_trace()

print(mesh_properties)

