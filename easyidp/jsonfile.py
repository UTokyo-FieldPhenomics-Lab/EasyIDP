import json
import os
import numpy as np
import geojson
import pyproj

from rich.table import Table
from rich.box import HORIZONTALS
from rich import print

class MyEncoder(json.JSONEncoder):
    # The original json package doesn't compatible to numpy object, add this compatible encoder to it.
    # usage: json.dump(..., cls=MyEncoder)
    # references: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    
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
    """Read json file to python dict.

    Parameters
    ----------
    json_path : str
        The path to json file

    Returns
    -------
    dict

    Example
    -------

    Data prepare:

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

    Use this function:

    .. code-block:: python

        >>> out = idp.jsonfile.read_json(test_data.json.for_read_json)
        >>> out
        {'test': {'rua': [[12, 34], [45, 56]]}, 'hua': [34, 34.567]}

    """
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
            return data
    else:
        raise FileNotFoundError(f"Could not locate the given json file [{json_path}]")
    
def _check_geojson_format(geojson_path):
    if '.geojson' not in str(geojson_path):
        raise TypeError(f"The input file format should be *.geojson, not [.{str(geojson_path).split('.')[-1]}]")

    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)

    # Only the FeatureCollection geojson type is supported.
    if geojson_data.type != 'FeatureCollection':
        raise TypeError(f'A `FeatureCollection` type geojson is expected, current type is `{geojson_data.type}`')
    
    if 'features' not in geojson_data.keys():
        raise TypeError('geojson does not have features properties')
    
    return geojson_data
    
def read_geojson(geojson_path):
    """Read geojson file to python dict

    Parameters
    ----------
    geojson_path : str
        The path to geojson file

    Returns
    -------
    dict
        .. code-block:: python

            {
                'crs': pyproj.CRS, 
                'geometry': [ 
                    {'coordiante': [[[...]]], 'type': '...'}, 
                    ... 
                ], 
                'property': [ 
                    {key1: v1, key2: v2},
                    ... 
                ]
            }
    
    Example
    -------
    Data prepare:

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

    Use this function:

    .. code-block:: python

        >>> out = idp.jsonfile.read_json(test_data.json.geojson_soy)
        >>> out
        {'crs': <Projected CRS: EPSG:6677>
            Name: JGD2011 / Japan Plane Rectangular CS IX
            Axis Info [cartesian]:
            - X[north]: Northing (metre)
            - Y[east]: Easting (metre)
            Area of Use:
            - name: Japan - onshore - Honshu - Tokyo-to.
            - bounds: (138.4, 29.31, 141.11, 37.98)
            Coordinate Operation:
            - name: Japan Plane Rectangular CS zone IX
            - method: Transverse Mercator
            Datum: Japanese Geodetic Datum 2011
            - Ellipsoid: GRS 1980
            - Prime Meridian: Greenwich,
         'geometry': [
            { 
                "coordinates": [[[-26384.952573, -28870.678514], ..., [-26384.952573, -28870.678514]]], 
                "type": "Polygon" 
            },
            ...
         ],
         'property': [
            {'FID': 65,
             '試験区': 'SubBlk 2b',
             'ID': 0,
             '除草剤': '有',
             'plotName': 'Enrei-10',
             'lineNum': 1},
            ...
         ]
        }

    """
    out_dict = {'crs': None, 'geometry': [], 'property':[]}

    geojson_data = _check_geojson_format(geojson_path)
    
    if 'crs' in geojson_data.keys():
        # ensure it has correct format for CRS info
        if  'properties' in geojson_data.crs and \
            'name'       in geojson_data.crs['properties']:
            out_dict['crs'] = pyproj.CRS.from_string(geojson_data.crs['properties']['name'])
        else:
            crs_template = {
                "type": "name",
                "properties": {
                    "name": "EPSG:xxxx"
                }
            }
            print(f'geojson does not have standard CRS properties like:\n{crs_template}'
                  f'\nBut the following is obtained:\n{geojson_data.crs}')
    else:
        print(f'[json][geojson] geojson does not have CRS properties')

    for f in geojson_data.features:
        out_dict['geometry'].append(f['geometry'])
        out_dict['property'].append(f['properties'])
    
    return out_dict

def show_geojson_fields(geojson_path):
    """
    Show geojson properties data, for better setting ``name_field`` of :py:obj:`read_geojson <easyidp.roi.ROI.read_geojson>`

    Parameters
    ----------
    geojson_path : : str
        the file path of \*.geojson

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> idp.jsonfile.show_geojson_fields(test_data.json.geojson_soy)
                          Properties of /Users/hwang/Library/Application                   
             Support/easyidp.data/data_for_tests/json_test/2023_soybean_field.geojson      
        ────────────────────────────────────────────────────────────────────────────────── 
         [-1]   [0] FID   [1] 試験区   [2] ID   [3] 除草剤    [4] plotName    [5] lineNum  
        ────────────────────────────────────────────────────────────────────────────────── 
            0     65      SubBlk 2b      0          有          Enrei-10           1       
            1     97      SubBlk 2b      0          有          Enrei-20           1       
            2     147     SubBlk 2b      0          有        Nakasenri-10         1       
          ...     ...        ...        ...        ...            ...             ...      
          257     259       SB 0a        0                   Tachinagaha-10        3       
          258      4        SB 0a        0                   Fukuyutaka-10         3       
          259      1      SubBlk 2a      0          無          Enrei-20           1       
        ──────────────────────────────────────────────────────────────────────────────────
        
    """
    geojson_data = _check_geojson_format(geojson_path)

    table = Table(title=f"Properties of {geojson_path}", box=HORIZONTALS)

    table.add_column("[-1]", justify='right')
    for i, c in enumerate(geojson_data.features[0]['properties'].keys()):
        table.add_column(f"[{i}] {c}", justify='center')

    row_num = len(geojson_data.features)
    col_num = len(geojson_data.features[0]['properties'].keys())

    if row_num >= 6:
        show_idx_top = [0, 1, 2]
        show_idx_btm = [-3, -2, -1]

        for i in show_idx_top:
            row = [str(i)] + [str(k) for k in geojson_data.features[i]['properties'].values()]
            table.add_row(*row)

        omit_row = ['...'] * (col_num + 1)
        table.add_row(*omit_row)

        for i in show_idx_btm:
            row = [str(row_num + i)] + [str(k) for k in geojson_data.features[i]['properties'].values()]
            table.add_row(*row)

    else:
        # print all without omit
        show_idx = list(range(row_num))

        for i in show_idx:
            row = [str(i)] + list(geojson_data.features[i]['properties'].values())
            table.add_row(*row)

    print(table)


def dict2json(data_dict, json_path, indent=None, encoding='utf-8'):
    """Save dict object to the same structure json file
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json`` 
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> a = {"test": {"rua": np.asarray([[12, 34], [45, 56]])}, "hua":[np.int32(34), np.float64(34.567)]}
        >>> idp.jsonfile.dict2json(a, "/path/to/save/json_file.json")

    .. note:: 

        Dict without indient:

        .. code-block:: python

            >>> print(json.dumps(data), indent=0)
            {"age": 4, "name": "niuniuche", "attribute": "toy"}

        Dict with 4 space as indient:

        .. code-block:: python

            >>> print(json.dumps(data,indent=4))
            {
                "age": 4,
                "name": "niuniuche",
                "attribute": "toy"
            }

    See also
    --------
    easyidp.jsonfile.write_json, easyidp.jsonfile.save_json
    
    """
    json_path = str(json_path)
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w', encoding=encoding) as result_file:
            json.dump(data_dict, result_file, ensure_ascii=False, cls=MyEncoder, indent=indent)

            # print(f'Save Json file -> {os.path.abspath(json_path)}')


def write_json(data_dict, json_path, indent=0, encoding='utf-8'):
    """Save dict to the same structure json file, a function wrapper for :func:`dict2json`
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json``
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file


    See also
    --------
    easyidp.jsonfile.dict2json

    """
    dict2json(data_dict, json_path, indent, encoding)


def save_json(data_dict, json_path, indent=0, encoding='utf-8'):
    """Save dict to the same structure json file, a function wrapper for :func:`dict2json`
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json``
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file

    See also
    --------
    easyidp.jsonfile.dict2json

    """
    dict2json(data_dict, json_path, indent, encoding)

# just copied from previous `caas_lite.py`, haven't modified yet
def _to_labelme_json(grid_tagged, json_folder, minimize=True):
    """Save the tagged shp polygon crop result to json file, for deeplearing use

    Parameters
    ----------
    grid_tagged : pandas.DataFrame
        the output of self.dataframe_add_shp_tags()
        The 4 column dataframe shows in this function introduction, sorted by "grid_name"
    json_folder : str
        the folder or path to save those json files
    minimize : bool
        whether create a json without space

    Notes
    -----
    The labelme json file has the following structure:

    .. code-block:: json

        {
            "version": "4.5.6",  # the Labelme.exe version, optional
            "flags": {},
            "imagePath": "xxxx.tiff",
            "imageHeight": 1000,
            "imageWidth": 1000,
            "imageData": null,
            "shapes": [{ }, { }, { }]
        }      

    for each ``{}`` items in "shapes":

    .. code-block:: json

        {
            "label": "field",
            "group_id": null,
            "shape_type": "polygon",
            "flags": {},
            "points": [[x1, y1], [x2, y2], [x3, y3]]  # with or without the first point
        }

    Example of ``grid_tagged`` 

    +------------------+----------+---------------------+-------+
    |     grid_name    | dict_key |    polygon_list     |  tag  |
    +==================+==========+=====================+=======+
    | 'grid_x1_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x1_y1.tif' | 'key2'   | [poly1, poly2, ...] | crops |
    +------------------+----------+---------------------+-------+
    | 'grid_x1_y2.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x2_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x3_y2.tif' | 'key2'   | [poly1, poly2, ...] | crops |
    +------------------+----------+---------------------+-------+
    |      ...         |   ...    |         ...         |  ...  | 
    +------------------+----------+---------------------+-------+

    Example of ``minimize``

    - True

      .. code-block:: json

        {"name":"lucy","sex":"boy"}
    
    - False

      .. code-block:: json

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