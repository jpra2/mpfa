import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from datamanager.mesh_data import MeshData
from definitions import defpaths
import numpy as np
import pandas as pd




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
    k = 20 # Number of clusters
    number_of_iterations = 100

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
        print(test)
        if test < kmeans_tol:
            stop = True

        epsilon_0 = epsilon[i]
        
    import pdb; pdb.set_trace()
    
    gids = cluster['centroid number'].to_numpy().astype(int) - 1
    cents = np.zeros((len(gids), 2))
    cents[:,0] = centroids['x'].to_numpy()
    cents[:,1] = centroids['y'].to_numpy()
    return gids, cents

def create_partition(level, x_centroids, y_centroids, faces_adjacencies):
    gid_name = 'gid_' + str(level)
    faces_adjacencies_coarse_name = 'faces_adjacencies_' + str(level)
    gids, coarse_centroids = create_partition_kmeans(x_centroids, y_centroids)



    # mesh_data = MeshData(mesh_path=mesh_path)
    # mesh_data.create_tag(gid_name, data_type='int')
    # mesh_data.insert_tag_data(gid_name, gids, elements_type='faces', elements_array=mesh_properties.faces)
    # mesh_data.export_all_elements_type_to_vtk(export_name='test_gids', element_type='faces')

    def create_faces_adjacencies(gids, faces_adjacencies):
        adjacencies = []
        all_adj = faces_adjacencies
        adj_test = (all_adj[:,0] != -1) & (all_adj[:, 1] != -1)
        

        pass
        

    mesh_properties.insert_data({
        gid_name: gids
    })

    mesh_properties.export_data()

def create_dual(level, x_centroids, y_centroids, adjacencies, gid_level):
    dual_name = 'dual_' + str(level)
    gids = np.arange(len(x_centroids))




    def create_vertices():
        pass

    def create_edges():
        pass

mesh_properties = create_initial_mesh_properties(mesh_path, mesh_name)

create_partition(
    1, 
    mesh_properties.faces_centroids[:,0], 
    mesh_properties.faces_centroids[:,1], 
    mesh_properties.faces_adj_by_edges
)

# mesh_properties = load_mesh_properties(mesh_name)


