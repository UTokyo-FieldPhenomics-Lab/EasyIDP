import json
import os
import numpy as np

class MyEncoder(json.JSONEncoder):
    """
    references: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def dict2json(given_dict, json_path):
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w') as result_file:
            json.dump(given_dict, result_file, cls=MyEncoder)

            print(f'[io][json] Save Json file -> {os.path.abspath(json_path)}')