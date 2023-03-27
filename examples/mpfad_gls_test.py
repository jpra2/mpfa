import os
from datamanager.meshmanager import CreateMeshProperties, MeshProperty, create_initial_mesh_properties, load_mesh_properties
from datamanager.mesh_data import MeshData
from definitions import defpaths
import numpy as np
import pandas as pd
from mpfa.lsds.gls_weight_2d import CalculateGlsWeight2D, Weight

mesh_path = os.path.join(defpaths.mesh, '2d_unstructured.msh')
mesh_name = 'mpfad_gls_test'
weight_name = 'weights_gls'

mesh_properties = create_initial_mesh_properties(mesh_path, mesh_name)

weight = Weight()
weight.insert_name(name=weight_name)

print(weight.class_name())
print(weight.class_path)

print(mesh_properties)