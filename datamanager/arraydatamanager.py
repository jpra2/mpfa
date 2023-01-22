import numpy as np
import copy
from pack_errors import errors
import os
from definitions import defpaths
import pandas as pd

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

class ArrayDataManager:
    
    def __init__(self, name=''):
        self.name = name
        self._data = dict()
    
    def insert_data(self, data):
        self._data.update(data)
    
    def export(self):
        np.savez(self.name, **self._data)
        print(f'\n{self.name} saved\n')
    
    def load(self):
        arq = np.load(self.name, allow_pickle=True)

        for name, variable in arq.items():
            self._data[name] = variable

        print(f'\n{self.name} loaded\n')
    
    def get_data_from_load(self):
        self.load()
        return copy.deepcopy(self._data)
    

class SuperArrayManager:
    
    def insert_name(self, name=''):
        self.__dict__['name'] = np.array([name])
    
    @property
    def class_name(self):
        return self.__class__.__name__    
    
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
    
    def __setattr__(self, name, value):
        raise Exception("It is read only! Use the 'insert_data' class function'")   
    
    @property
    def class_path(self):
        return os.path.join(defpaths.flying, self.class_name + '_' +  self.name[0] + '.npz')
    
    def export_data(self):
        manager = ArrayDataManager(self.class_path)
        manager.insert_data(self.__dict__)
        manager.export()

    def load_data(self):
        manager = ArrayDataManager(self.class_path)
        self.insert_data(manager.get_data_from_load())