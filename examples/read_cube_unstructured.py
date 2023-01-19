import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty
from definitions import defpaths
from utils import calculate_face_properties


###
## run the code examples.create_cube_unstructured for generate mesh
###

mesh_path = os.path.join(defpaths.mesh, 'cube_unstructured.msh')
mesh_create = CreateMeshProperties()
mesh_name = 'cube_unstructured_test'
mesh_create.initialize(mesh_path=mesh_path, mesh_name=mesh_name)
mesh_properties: MeshProperty = mesh_create.create_3d_mesh_data()

faces_area, faces_normal_unitary  = calculate_face_properties.define_normal_and_area(mesh_properties.faces, mesh_properties.nodes_of_faces, mesh_properties.nodes_centroids)

mesh_properties.insert_data(
    {
        'faces_area': faces_area,
        'unitary_normal_faces': faces_normal_unitary
    }
)

# mesh_properties.export_data()

print(dir(mesh_properties))
# import pdb; pdb.set_trace()





