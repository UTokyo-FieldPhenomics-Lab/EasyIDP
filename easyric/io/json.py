import json
import os

def dict2json(clip_dict, json_path):
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w') as result_file:
            json.dump(clip_dict, result_file)

            print(f'[io][json] Save Json file -> {os.path.abspath(json_path)}')