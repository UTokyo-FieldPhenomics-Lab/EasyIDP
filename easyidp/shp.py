import os
import pyproj
import shapefile
import numpy as np
from tabulate import tabulate
from tqdm import tqdm


def read_proj(prj_path):
    """
    read *.prj file to pyproj object
    
    Parameters
    ----------
    prj_path : str
        the file path of shp *.prj
    
    Returns
    -------
    proj : the pyproj object
    """
    with open(prj_path, 'r') as f:
        wkt_string = f.readline()

    proj = pyproj.CRS.from_wkt(wkt_string)
    
    if proj.name == 'WGS 84':
        proj = pyproj.CRS.from_epsg(4326)

    return proj


def show_shp_fields(shp_path, encoding="utf-8", show_num=5):
    """
    Read shp field data to pandas.DataFrame, for further json metadata usage
    
    Parameters
    ----------
    shp_path : str
        the file path of *.shp
    encoding : str
        default is 'utf-8', however, or some chinese characters, 'gbk' is required
    """
    shp = shapefile.Reader(shp_path, encoding=encoding)

    # read shp file fields
    shp_fields = _get_field_key(shp)

    header_str_list = [f"[{v}] {k}" for k, v in shp_fields.items()]

    content_str_list = []
    if len(shp.records()) > show_num * 2:
        # first {show_num} rows
        for i in range(0, show_num):
            content_str_list.append(list(shp.records()[i]))

        # omitted row
        omit_list = ['...'] * len(shp.records()[i])
        content_str_list.append(omit_list)

        # last {show_num} rows
        for i in range(-show_num, 0):
            content_str_list.append(list(shp.records()[i]))
    else:
        # print all rows without omit
        for i in shp.records():
            content_str_list.append(list(i))

    table_str = tabulate(content_str_list, header_str_list, tablefmt="github")
    print(table_str)


def read_shp(shp_path, shp_proj=None, name_field=None, include_title=False, encoding='utf-8', return_proj=False):
    """
    read shp file to python numpy object
    
    Parameters
    ----------
    shp_path : str
        the file path of *.shp
    name_field : str or int or list[ str|int ], 
        the id or name of shp file fields as output dictionary keys
    shp_proj : str | pyproj object
        default None, 
            -> will read automatically from prj file with the same name of shp filename, 
        or give manually
            -> read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)  # default WGS 84 with longitude and latitude
            -> read_shp(..., shp_proj=r'path/to/{shp_name}.prj', ...)
    encoding : str
        default is 'utf-8', however, or some chinese characters, 'gbk' is required
    
    Returns
    -------
    shp_dict: dict, the dictionary with read numpy polygon coordinates
                {'id1': np.array([[x1,y1],[x2,y2],...]),
                'id2': np.array([[x1,y1],[x2,y2],...]),...}
    """
    #####################################
    # check projection coordinate first #
    #####################################
    if shp_proj is None:
        prj_path = shp_path[:-4] + '.prj'
        if os.path.exists(prj_path):
            shp_proj = read_proj(prj_path)
        else:
            raise ValueError(f"Unable to find the proj coordinate info [{prj_path}], please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`")
    # or give a prj file path
    elif isinstance(shp_proj, str) and shp_proj[-4:]=='.prj' and os.path.exists(shp_proj):
        shp_proj = read_proj(shp_proj)
    # or give a CRS projection object
    elif isinstance(shp_proj, pyproj.CRS):
        pass
    else:
        raise ValueError(f"Unable to find the projection coordinate, please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`")

    print(f'[shp][proj] Use projection [{shp_proj.name}] for loaded shapefile [{os.path.basename(shp_path)}]')

    # read shapefile
    shp = shapefile.Reader(shp_path, encoding=encoding)
    
    # read shp file fields (headers)
    shp_fields = _get_field_key(shp)

    ########################
    # read shp coordinates #
    ########################
    shp_dict = {}

    ### do not put it into the following loop, save calculation time.
    if isinstance(name_field, list):
        field_id = [_find_name_related_int_id(shp_fields, nf) for nf in name_field]
    else:
        field_id = _find_name_related_int_id(shp_fields, name_field)

    pbar = tqdm(shp.shapes(), desc=f"[shp] read shp [{os.path.basename(shp_path)}]")
    for i, shape in enumerate(pbar):
        # convert dict_key name string by given name_field
        if isinstance(field_id, list):
            plot_name = ""
            for j, fid in enumerate(field_id):
                if include_title:
                    plot_name += f"{_find_key(shp_fields, fid)}_{shp.records()[i][fid]}"
                else:
                    plot_name += f"{shp.records()[i][fid]}"

                # not adding the last key A_B_C_ --> A_B_C
                if j < len(field_id)-1:
                    plot_name += "_"
        elif field_id is None:
            if include_title:
                plot_name = f"line_{i}"
            else:
                plot_name = f"{i}"
        else:
            if include_title:
                plot_name = f"{_find_key(shp_fields, field_id)}_{shp.records()[i][field_id]}"
            else:
                plot_name = f"{shp.records()[i][field_id]}"

        plot_name = plot_name.replace(r'/', '_')
        plot_name = plot_name.replace(r'\\', '_')

        ##################################
        # get the shape coordinate value #
        ##################################
        coord_np = np.asarray(shape.points)
        # check if the last point == first point
        if (coord_np[0, :] != coord_np[-1, :]).all():
            # otherwise duplicate first point to last point to fit the polygon definition
            coord_np = np.append(coord_np, coord_np[0,:][None,:], axis = 0)

        # if unit is degrees, exchange x and y
        x_unit = shp_proj.coordinate_system.axis_list[0].unit_name
        y_unit = shp_proj.coordinate_system.axis_list[1].unit_name
        if x_unit == "degree" and y_unit == "degree": 
            coord_np = np.flip(coord_np, axis=1)
        # ----------- Notes --------------
        # when shp unit is (degrees) latitiude and longitude
        #     the default order is (lon, lat) --> (y, x)
        # in easyidp, numpy, and pyproj coordiate system, needs (lat, lon), so need to revert
        #   
        #   ↑                 y         
        #   N   O------------->  
        #       |  ________      
        #       | |        |     
        #       | |  img   |     
        #       | |________|     
        #     x v                
        #
        # however, if unit is meter (e.g. UTM Zone), the order doesn't need to be changed
        # ---------------------------------

        # check if has duplicated key, otherwise will cause override
        if plot_name in shp_dict.keys():
            raise KeyError(f"Meet with duplicated key [{plot_name}] for current shapefile, please specify another `name_field` from {shp_fields} or simple leave it blank `name_field=None`")

        shp_dict[plot_name] = coord_np

    if return_proj:
        return shp_dict, shp_proj
    else:
        return shp_dict


def convert_proj(shp_dict, origin_proj, target_proj):
    """ 
    Provide the geo coordinate transfrom based on pyproj package

    Parameters
    ----------
    shp_dict : dict
        the output of read_shp() function
    shp_proj : pyproj object
        the hidden output of read_shp(..., return_proj=True)
    target_proj : str | pyproj object
        e.g. 
        [1] pyproj.CRS.from_epsg(4326)  # default WGS 84 longitude latitude
        [2] r'path/to/{shp_name}.prj',
    """
    transformer = pyproj.Transformer.from_proj(origin_proj, target_proj)
    trans_dict = {}
    for k, coord_np in shp_dict.items():
        transformed = transformer.transform(coord_np[:, 0], coord_np[:, 1])
        coord_np = np.asarray(transformed).T

        # judge if has inf value, means convert fail
        if True in np.isinf(coord_np):
            raise ValueError(
                f'Fail to convert points from "{origin_proj.name}" to '
                f'"{target_proj.name}"(dsm projection), '
                f'this may caused by the uncertainty of .prj file strings, '
                f'please check the coordinate manually via QGIS Layer Infomation, '
                f'get the EPGS code, and specify the function argument'
                f'read_shp2d(..., given_proj=pyproj.CRS.from_epsg(xxxx))')
        trans_dict[k] = coord_np

    return trans_dict


def _get_field_key(shp):
    """
    Convert shapefile header {"Column": int_id}

    Parameters
    ----------
    shp : shapefile.Reader object
        shp = shapefile.Reader(shp_path, encoding=encoding)

    Returns
    -------
    shp_fields : dict
        Format: {"Column": int_id}
        Exmaple: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
    """
    shp_fields = {}
    f_count = 0
    for l in shp.fields:
        if isinstance(l, list):
            # the fields 0 -> delection flags, and is a tuple type, ignore this tag
            # [('DeletionFlag', 'C', 1, 0),
            #  ['ID', 'C', 36, 0],
            #  ['MASSIFID', 'C', 19, 0],
            #  ['CROPTYPE', 'C', 36, 0],
            #  ['CROPDATE', 'D', 8, 0],
            #  ['CROPAREA', 'N', 13, 5],
            #  ['ATTID', 'C', 36, 0]]
            shp_fields[l[0]] = f_count
            f_count += 1

    return shp_fields
    
    
def _find_name_related_int_id(shp_fields, name_field):
    """
    Inner function to get the number of given `name_field`.

    Parameters
    ----------
    shp_fields : dict
        the output of _get_field_key()
        Format: {"Column": int_id}
        Exmaple: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
    name_field : str or int or list[ str|int ], 
        the id or name of shp file fields as output dictionary keys

    Returns
    -------
    field_id : int or list[ int ]
        e.g. 
        >>> a = {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
        >>> b = "ID"
        >>> _find_name_related_int_id(a, b)
        0
        >>> c = ["ID", "CROPTYPE"]
        >>> _find_name_related_int_id(a, b)
        [0, 2]
    """
    if name_field is None:
        field_id = None
    elif isinstance(name_field, int):
        if name_field >= len(shp_fields):
            raise IndexError(f'Int key [{name_field}] is outside the number of fields {shp_fields}')
        field_id = name_field
    elif isinstance(name_field, str):
        if name_field not in shp_fields.keys():
            raise KeyError(f'Can not find key {name_field} in {shp_fields}')
        field_id = shp_fields[name_field]
    else:
        raise KeyError(f'Can not find key {name_field} in {shp_fields}')
    
    return field_id


def _find_key(mydict, value):
    """
    a simple function to using dict value to find key
    e.g. 
    >>> mydict = {"a": 233, "b": 456}
    >>> _find_key(mydict, 233)
    "a"
    """
    return list(mydict.keys())[value]
