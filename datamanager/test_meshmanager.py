import unittest
import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from definitions import defpaths

class TestCreateMeshProperty(unittest.TestCase):
    
    def setUp(self) -> None:
        mesh_path = defpaths.mesh_test_path
        self.mesh_name = 'mesh_test'
        self.mesh_create = CreateMeshProperties()
        self.mesh_create.initialize(mesh_path=mesh_path, mesh_name=self.mesh_name)
        
    
    def test_create_mesh(self):
        mesh_properties: MeshProperty = self.mesh_create.create_mesh_data()
        self.assertTrue(os.path.exists(mesh_properties.mesh_path), 'Erro test_create_mesh')
    
    def test_load_mesh(self):
        mesh_properties = MeshProperty()
        mesh_properties.insert_mesh_name(self.mesh_name)
        mesh_properties.load_data()
        self.assertEqual(mesh_properties.mesh_name, self.mesh_name, 'Erro test_load_mesh')

# if __name__ == '__name__':
#     unittest.main()