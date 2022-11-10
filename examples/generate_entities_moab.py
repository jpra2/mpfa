import os
from datamanager.meshmanager import CreateMeshProperties
from definitions import defpaths


###
## run the code examples.create_cube_unstructured for generate mesh
###

# mesh_path = os.path.join(defpaths.mesh, 'cube_structured2.msh')
mesh_path = os.path.join(defpaths.mesh, 'cube_unstructured.msh')
mesh_create = CreateMeshProperties()
mesh_name = 'cube_structured2_test'
mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
mesh_create.init_mesh()
mesh_create.init_mesh_entities()
mesh_create.export_data_msh()