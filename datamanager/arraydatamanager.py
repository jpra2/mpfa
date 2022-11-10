import numpy as np
import copy

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
    
     