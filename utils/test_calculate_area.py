import unittest
from utils import calculate_face_properties as calculate_area
import numpy as np

class TestCalculateArea(unittest.TestCase):
    
    def setUp(self) -> None:
        self. vtest = np.array([
                        [  -1,    0, 0],
                        [   0,    1, 0],
                        [-0.5, -0.5, 0],
                        [   1,    0, 0],
                        [ 0.5, -0.5, 0]
                    ])
        
    def test_order_vertices(self):
        # idtest = np.array([874, 873, 870, 871, 872])
        idtest = np.array([874, 872, 870, 871, 873])
        
        indices = np.array([870, 871, 872, 873, 874])
        new_ids = calculate_area.sort_radial_sweep(self.vtest, indices)
        self.assertTrue(np.allclose(new_ids, idtest), 'ERRO test_order_vertices')
    
    def test_calculate_area_poly_3d(self):
        
        
        indices = np.arange(len(self.vtest))
        new_ids = calculate_area.sort_radial_sweep(self.vtest, indices)
        area = calculate_area.polygon_area(
            self.vtest[new_ids]
        )
        area2 = calculate_area.polygon_area(self.vtest[new_ids] + 5)
        
        self.assertEqual(area, 1.75)
        self.assertEqual(area, area2)
    
    