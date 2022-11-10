import os
from datamanager.meshmanager import CreateMeshProperties
from definitions import defpaths


###
## run the code examples.create_cube_unstructured for generate mesh
###

mesh_path = os.path.join(defpaths.mesh, 'cube_structured.msh')
mesh_create = CreateMeshProperties()
mesh_name = 'cube_structured_test'
mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
mesh_properties = mesh_create.create_mesh_data()





