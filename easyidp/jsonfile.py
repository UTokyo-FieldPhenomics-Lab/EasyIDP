import json
import os
import numpy as np

class MyEncoder(json.JSONEncoder):
    """
    The original json package doesn't compatible to numpy object, add this compatible encoder to it.
    usage: json.dump(..., cls=MyEncoder)
    
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

def read_json(json_path):
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
            return data
    else:
        raise FileNotFoundError(f"Could not locate the given json file [{json_path}]")

def dict2json(data_dict, json_path, indent=None, encoding='utf-8'):
    """
    Convert dict to the same structure json file
    
    Parameters
    ----------
    clip_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. "D:/xxx/xxxx/save.json"
    indent : int | None
        whether save "readable" json with indent, default 0 without indent

        example:
        >>> print(json.dumps(data), indent=0)
        {"age": 4, "name": "niuniuche", "attribute": "toy"}
        >>> print(json.dumps(data,indent=4))
        {
            "age": 4,
            "name": "niuniuche",
            "attribute": "toy"
        }
    encoding : str
        the encoding type of output file
    """
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w', encoding=encoding) as result_file:
            json.dump(data_dict, result_file, ensure_ascii=False, cls=MyEncoder, indent=indent)

            # print(f'Save Json file -> {os.path.abspath(json_path)}')


def write_json(data_dict, json_path, indent=0, encoding='utf-8'):
    dict2json(data_dict, json_path, indent, encoding)

def save_json(data_dict, json_path, indent=0, encoding='utf-8'):
    dict2json(data_dict, json_path, indent, encoding)

# just copied from previous `caas_lite.py`, haven't modified yet
def _to_labelme_json(grid_tagged, json_folder, minimize=True):
    """
    Save the tagged shp polygon clip result to json file, for deeplearing use
        The single json file has the following structure:
            {
                "version": "4.5.6",  # the Labelme.exe version, optional
                "flags": {},
                "imagePath": "xxxx.tiff",
                "imageHeight": 1000,
                "imageWidth": 1000,
                "imageData": null,
                "shapes": [{ }, { }, { }]
            }             |    |    |
        for each {} items in shapes:
            {
                "label": "field",
                "group_id": null,
                "shape_type": "polygon",
                "flags": {},
                "points": [[x1, y1], [x2, y2], [x3, y3]]  # with or without the first point
            }
    
    Parameters
    ----------
    grid_tagged: pandas.DataFrame
        the output of self.dataframe_add_shp_tags()
        The 4 column dataframe shows in this function introduction, sorted by "grid_name"
        +------------------+----------+---------------------+-------+
        |     grid_name    | dict_key |    polygon_list     |  tag  |
        |------------------|----------|---------------------|-------|
        | 'grid_x1_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
        | 'grid_x1_y1.tif' | 'key2'   | [poly1, poly2, ...] | crops |
        | 'grid_x1_y2.tif' | 'key1'   | [poly1, poly2, ...] | field |
        | 'grid_x2_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
        | 'grid_x3_y2.tif' | 'key2'   | [poly1, poly2, ...] | crops |
        |      ...         |   ...    |         ...         |  ...  | 
        +------------------+----------+---------------------+-------+
    json_folder: str
        the folder or path to save those json files
    minimize: bool
        True:  {"name":"lucy","sex":"boy"}
        False：
                {
                    "name":"lucy",
                    "sex":"boy"
                } 
    """
    total_dict = {}
    for i in range(len(grid_tagged)):
        # todo modify here
        img_name = grid_tagged.loc[i]['grid_name']
        poly = grid_tagged.loc[i]['polygon_list']
        tag = grid_tagged.loc[i]['tag']
        
        if img_name not in total_dict.keys():
            single_dict = {"version": "4.5.6", 
                            "flags": {},
                            "imagePath": img_name,
                            "imageHeight": 1000,
                            "imageWidth": 1000,
                            "imageData": None,
                            "shapes": []}
        else:
            single_dict = total_dict[img_name]
            
        for item in poly:
            single_item = {"label": tag,
                            "group_id": None,   # json null = python None
                            "shape_type": "polygon",
                            "flags": {},
                            "points": item.tolist()}
            single_dict['shapes'].append(single_item)
            
        total_dict[img_name] = single_dict
        
    # after iter all items
    for k, d in total_dict.items():
        json_name = k.replace('.tif', '.json')
        if minimize:
            dict2json(d, os.path.join(json_folder, json_name))
        else:
            dict2json(d, os.path.join(json_folder, json_name), indent=2)