from datamanager.generic_properties import RockProperties
import numpy as np

case_to_run = 'rock_prop_test'
rock_properties = RockProperties()
rock_properties.insert_name(case_to_run)

print(rock_properties.class_name)
print(rock_properties.class_path)
print(rock_properties.__dict__)

rock_properties.insert_data({
    'permeability': np.array([10, 11, 12])
})

import pdb; pdb.set_trace()