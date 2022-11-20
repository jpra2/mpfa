import os
from datamanager.meshmanager import CreateMeshProperties
from definitions import defpaths

mesh_path = os.path.join(defpaths.mesh, 'cube10_salome.gmsh')
mesh_create = CreateMeshProperties()
mesh_name = 'cube_structured_test_salome'
mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
mesh_properties = mesh_create.create_mesh_data()