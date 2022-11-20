# from impress.preprocessor.meshHandle.finescaleMesh import FineScaleMesh as msh
from pymoab import core, types, rng, topo_util
from pymoab import skinner as sk
from pymoab.scd import ScdInterface
from pymoab.hcoord import HomCoord
import numpy as np
import pandas as pd
import copy
from definitions import defpaths
import os
from datamanager.arraydatamanager import ArrayDataManager
from pack_errors import errors

def test_array_instance(data):
    if isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError

def test_str_instance(data):
    if isinstance(data, str):
        pass
    else:
        raise TypeError

class MeshProperty:
    
    def insert_mesh_name(self, name=''):
        self.__dict__['mesh_name'] = name
    
    def insert_data(self, data: dict):
        """data is a dictionary with str keys and np.ndarray values

        Args:
            data (_type_): dict
        """
        names = list(data.keys())
        values = list(data.values())
        
        a = [test_array_instance(_) for _ in values]
        a = [test_str_instance(_) for _ in names]
        
        names_series = pd.DataFrame({
            'names': names
        })
        
        names_data_self = np.array(list(self.__dict__.keys()))
        names_data_self = names_data_self[names_data_self != 'mesh_name']
        test = names_series.isin(names_data_self)
        if test.any().values[0]:
            names_in = names_series[test.values].values.flatten()
            raise errors.NameExistsError(f'The names: - {names_in} - exists in mesh properties')
        
        
        self.__dict__.update(data)
        # self._data.update(data)
        # self.__dict__['mesh_name'] = data['mesh_name'][0]
    
    def __setattr__(self, name, value):
        raise Exception("It is read only!")      
    
    # @property
    # def volumes(self):
    #     return self._data['volumes']
    
    # @property
    # def faces(self):
    #     return self._data['faces']
    
    # @property
    # def nodes(self):
    #     return self._data['nodes']

    # @property
    # def nodes_of_faces(self):
    #     return self._data['nodes_of_faces']

    # @property
    # def volumes_adjacencies_by_faces(self):
    #     return self._data['volumes_adj_by_faces']
        
    # @property
    # def nodes_centroids(self):
    #     return self._data['nodes_centroids']

    # @property
    # def internal_faces(self):
    #     return self.faces[self._data['bool_internal_faces']]
    
    # @property
    # def boundary_faces(self):
    #     return np.setdiff1d(self.faces, self.internal_faces)
    
    # @property
    # def nodes_of_volumes(self):
    #     return self._data['nodes_of_volumes']
    
    # @property
    # def faces_of_volumes(self):
    #     return self._data['faces_of_volumes']
    
    @property
    def mesh_path(self):
        return os.path.join(defpaths.flying, 'mesh_property_' + self.mesh_name[0] + '.npz')
    
    def export_data(self):
        manager = ArrayDataManager(self.mesh_path)
        manager.insert_data(self.__dict__)
        manager.export()

    def load_data(self):
        manager = ArrayDataManager(self.mesh_path)
        self.insert_data(manager.get_data_from_load())
        
    

class CreateMeshProperties():

    '''
        Create mesh properties using pymoab
    '''

    def initialize(self, mesh_path='', mesh_name=''):

        self.mesh_path = mesh_path
        self.all_volumes = None
        self.all_faces = None
        self.all_nodes = None
        self.all_edges = None
        self.mb: core.Core = None
        self.mtu: topo_util.MeshTopoUtil = None
        self.root_set = None
        self.data = dict()
        self.mesh_name = mesh_name

    def _init_mesh(self):
        mb = core.Core()
        scd = ScdInterface(mb)
        mtu = topo_util.MeshTopoUtil(mb)
        mb.load_file(self.mesh_path)
        root_set = mb.get_root_set()
        return mb, mtu, root_set

    def init_mesh(self):
        self.mb, self.mtu, self.root_set = self._init_mesh()

    def init_mesh_entities(self):
        self.all_volumes = self.mb.get_entities_by_dimension(0, 3)
        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)
        boundary_faces = self.mb.get_entities_by_dimension(0, 2)
        # edges = self.mb.get_entities_by_dimension(0, 1)
        
        # all_entities = self.mb.get_entities_by_handle(self.root_set)
        
        # all_entities = np.setdiff1d(all_entities, self.all_volumes)
        # all_entities = np.setdiff1d(all_entities, self.all_nodes)
        # all_entities = np.setdiff1d(all_entities, faces)
        # all_entities = np.setdiff1d(all_entities, edges)
        
        # for ent in all_entities:
        #     ents = self.mb.get_entities_by_handle(ent)
        #     print(ents)
        
        
        # type_moab = self.mb.type_from_handle(self.root_set)
        all_moab_types = dir(types)
        
        self.dict_moab_types = dict()
        
        for tt in all_moab_types[0:-20]:
            
            # exec('print(types.' + tt + ')')
            # exec('respp[tt] = types.' + tt)
            exec('self.dict_moab_types[types.' + tt +'] = tt')
        
        
        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        # other_faces = []
        # for face in self.all_faces:
        #     tags = self.mb.tag_get_tags_on_entity(face)
        #     if len(tags) != 1:
        #         other_faces.append(face)
        
        # boundary_faces = np.setdiff1d(self.all_faces, other_faces)
        
        self.all_edges = self.mb.get_entities_by_dimension(0, 1)

    def _init_properties(self, faces, volumes, nodes):
        n_faces = len(faces)
        volumes_adj_by_faces = np.repeat(-1, n_faces*2).reshape((n_faces, 2)).astype(np.uint64)
        volumes_series = pd.DataFrame({
            'vol_ids': volumes
        }, index = np.array(self.all_volumes))

        nodes_series = pd.DataFrame({
            'nodes_ids': nodes
        }, index=self.all_nodes)
        
        faces_series = pd.DataFrame({
            'faces_ids': faces
        }, index=self.all_faces)

        # nodes_of_faces = np.repeat(-1, n_faces*4).reshape((n_faces, 4)).astype(np.uint64)
        nodes_of_faces = []
        faces_of_volumes = []
        nodes_of_volumes = []
        volumes_adj_by_nodes = []

        for i, face in enumerate(self.all_faces):
            volumes_adj_by_faces[i][:] = self.mtu.get_bridge_adjacencies(face, 2, 3)
            nodes_of_faces_elems = self.mtu.get_bridge_adjacencies(face, 2, 0) 
            nodes_of_faces.append(nodes_series.loc[nodes_of_faces_elems].to_numpy().flatten())
            
        for vol in self.all_volumes:
            faces_of_volumes_elements = self.mtu.get_bridge_adjacencies(vol, 3, 2)
            nodes_of_volumes_elements = self.mtu.get_bridge_adjacencies(vol, 3, 0)
            vols_by_nodes_elements = self.mtu.get_bridge_adjacencies(vol, 0, 3)
            faces_of_volumes_loc = faces_series.loc[faces_of_volumes_elements].to_numpy().flatten()
            nodes_of_volumes_loc = nodes_series.loc[nodes_of_volumes_elements].to_numpy().flatten()
            vols_by_nodes_loc = volumes_series.loc[vols_by_nodes_elements].to_numpy().flatten()
            faces_of_volumes.append(faces_of_volumes_loc)
            nodes_of_volumes.append(nodes_of_volumes_loc)
            volumes_adj_by_nodes.append(vols_by_nodes_loc)
        

        test = volumes_adj_by_faces[:, 0] == volumes_adj_by_faces[:, 1]
        volumes_adj_by_faces[:, 0] = volumes_series.loc[volumes_adj_by_faces[:, 0]].to_numpy().flatten()
        volumes_adj_by_faces[:, 1] = volumes_series.loc[volumes_adj_by_faces[:, 1]].to_numpy().flatten()
        volumes_adj_by_faces = volumes_adj_by_faces.astype(np.int64)
        volumes_adj_by_faces[test, 1] = -1
        
        nodes_of_faces = np.array(nodes_of_faces)
        faces_of_volumes = np.array(faces_of_volumes)
        nodes_of_volumes = np.array(nodes_of_volumes)
        volumes_adj_by_nodes = np.array(volumes_adj_by_nodes)
        
        bool_internal_faces = test

        return bool_internal_faces, volumes_adj_by_faces, nodes_of_faces, faces_of_volumes, nodes_of_volumes, volumes_adj_by_nodes

    def create_initial_array_properties(self):

        volumes = np.arange(len(self.all_volumes), dtype=int)
        faces = np.arange(len(self.all_faces), dtype=int)
        edges = np.arange(len(self.all_edges), dtype=int)
        nodes = np.arange(len(self.all_nodes), dtype=int)
        bool_internal_faces, volumes_adj_by_faces, nodes_of_faces, faces_of_volumes, nodes_of_volumes, volumes_adj_by_nodes =  self._init_properties(faces, volumes, nodes)

        nodes_centroids = np.array([self.mb.get_coords(node) for node in self.all_nodes])

        # av2 = np.array(self.all_volumes, np.uint64)
        #
        # vc = self.mtu.get_average_position(av2[0])

        self.data['volumes'] = volumes
        self.data['faces'] = faces
        self.data['edges'] = edges
        self.data['nodes'] = nodes
        self.data['bool_internal_faces'] = bool_internal_faces
        self.data['volumes_adj_by_faces'] = volumes_adj_by_faces
        self.data['volumes_adj_by_nodes'] = volumes_adj_by_nodes
        self.data['nodes_of_faces'] = nodes_of_faces
        self.data['nodes_centroids'] = nodes_centroids
        self.data['faces_of_volumes'] = faces_of_volumes
        self.data['nodes_of_volumes'] = nodes_of_volumes
        self.data['mesh_name'] = np.array([self.mesh_name])
    
    def export_data_msh(self):
        # import pdb; pdb.set_trace()
        tags = self.mb.tag_get_tags_on_entity(self.root_set)
        import pdb; pdb.set_trace()
        
        self.mb.write_file(os.path.join(defpaths.mesh, self.mesh_name) + '.msh')
        
    
    def create_mesh_data(self):
        self.init_mesh()
        self.init_mesh_entities()
        self.create_initial_array_properties()
        mesh_property = MeshProperty()
        mesh_property.insert_mesh_name(self.data['mesh_name'][0])
        mesh_property.insert_data(copy.deepcopy(self.data))
        mesh_property.export_data()
        return mesh_property
        




# if __name__ == '__main__':
#     mesh_path = 'mesh/80x80x1_ufce.h5m'
#     mesh_name = '80x80_ufce'
#     mesh_create = CreateMeshProperties()
#     mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
#     mesh_properties = mesh_create.create_mesh_data()
    
#     import pdb; pdb.set_trace()