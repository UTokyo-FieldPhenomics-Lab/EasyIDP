import os
import pyproj
import numpy as np
from shapely.geometry import Polygon, Point
from easyric.external import shapefile
from easyric.io.geotiff import point_query, mean_values, min_values, get_header


def read_proj(prj_path):
    with open(prj_path, 'r') as f:
        wkt_string = f.readline()

    proj = pyproj.CRS.from_wkt(wkt_string)

    if proj.name == 'WGS 84':
        proj = pyproj.CRS.from_epsg(4326)

    return proj


def _convert_shp_title(shp_fields, name_field):
    if name_field is None:
        field_id = None
    elif isinstance(name_field, int):
        field_id = name_field
    elif isinstance(name_field, str):
        field_id = shp_fields[name_field]
    else:
        raise KeyError(f'Can not find key {name_field} in {shp_fields}')
    
    return field_id


def _find_key(mydict, value):
    return list(mydict.keys())[value]


def read_shp2d(shp_path, shp_proj=None, geotiff_proj=None, name_field=None, title_include=False, encoding='utf-8'):
    # [Todo] write a test from name_field=None to list ["plot", "weed"] and [0,1]
    # [Todo] given a warning when the column has the same value (will cause overwride of dictionary key)
    shp = shapefile.Reader(shp_path, encoding=encoding)
    shp_dict = {}
    # read shp file fields
    shp_fields = {}
    #for i, l in enumerate(shp.fields):
    #    if isinstance(l, list):
    #        shp_fields[l[0]] = i
    fields = shp.fields[1:]
    shp_fields = {field[0]:i for i, field in enumerate(fields)}

    print(f'[io][shp][fields] Shp fields: {shp_fields}')

    # try to find current projection
    if shp_proj is None:
        prj_path = shp_path[:-4] + '.prj'
        if os.path.exists(prj_path):
            shp_proj = read_proj(shp_path[:-4] + '.prj')
            print(f'[io][shp][proj] find ESRI projection file {prj_path}, and successfully obtain projection '
                  f'{shp_proj.coordinate_system}')
        else:
            print(f'[io][shp][proj] could not find ESRI projection file {prj_path}, could not operate auto-convention, '
                  f'Please convert projection system manually.')

    total_num = len(shp.shapes())

    if isinstance(name_field, list):
        field_id = [_convert_shp_title(shp_fields, nf) for nf in name_field]
    else:
        field_id = _convert_shp_title(shp_fields, name_field)

    for i, shape in enumerate(shp.shapes()):
        if isinstance(field_id, list):
            plot_name = ""
            for j, fid in enumerate(field_id):
                if title_include:
                    plot_name += f"{_find_key(shp_fields, fid)}_{shp.records()[i][fid]}"
                else:
                    plot_name += f"{shp.records()[i][fid]}"

                if j < len(field_id)-1:
                    plot_name += "_"
        elif field_id is None:
            if title_include:
                plot_name = f"line_{i}"
            else:
                plot_name = f"{i}"
        else:
            if title_include:
                plot_name = f"{_find_key(shp_fields, field_id)}_{shp.records()[i][field_id]}"
            else:
                plot_name = f"{shp.records()[i][field_id]}"

        plot_name = plot_name.replace(r'/', '_')
        plot_name = plot_name.replace(r'\\', '_')

        coord_np = np.asarray(shape.points)

        # Old version: the pyshp package load seems get (lon, lat), however, the pyproj use (lat, lon), so need to revert
        # latest version:
        # when shp unit is (degrees) lat, lon, the order is reversed
        # however, if using epsg as unit (meter), the order doesn't need to be changed
        if coord_np.max() <= 180.0:
            coord_np = np.flip(coord_np, axis=1)

        if geotiff_proj is not None and shp_proj is not None and shp_proj.name != geotiff_proj.name:
            transformer = pyproj.Transformer.from_proj(shp_proj, geotiff_proj)
            transformed = transformer.transform(coord_np[:, 0], coord_np[:, 1])
            coord_np = np.asarray(transformed).T

            if True in np.isinf(coord_np):
                raise ValueError(f'Fail to convert points from "{shp_proj.name}"(shp projection) to '
                                 f'"{geotiff_proj.name}"(dsm projection), '
                                 f'this may caused by the uncertainty of .prj file strings, please check the coordinate '
                                 f'manually via QGIS Layer Infomation, get the EPGS code, and specify the function argument'
                                 f'read_shp2d(..., given_proj=pyproj.CRS.from_epsg(xxxx))')

        shp_dict[plot_name] = coord_np

        print(f"[io][shp][name] Plot {plot_name} loaded | {i+1}/{total_num}        ", end="\r")

    return shp_dict


def read_shp3d(shp_path, dsm_path, get_z_by='mean', get_z_buffer=0, shp_proj=None, geotiff_proj=None, geo_head=None, name_field=None, title_include=False, encoding='utf-8'):
    """[summary]

    Parameters
    ----------
    shp_path : str
        full shp_file directory
    dsm_path : str
        full dsm_file directory where want to extract height from
    get_z_by : str, optional
        ["local", "mean", "min", "max", "all"], by default 'mean'
        - "local" using the z value of where boundary points located, each point will get different z-values
                -> this will get a 3D curved mesh of ROI
        - "mean": using the mean value of boundary points closed part.
        - "min": 5th percentile mean height (the mean value of all pixels < 5th percentile)
        - "max": 95th percentile mean height (the mean value of all pixels > 95th percentile)
        - "all": using the mean value of whole DSM as the same z-value for all boundary points
                -> this will get a 2D plane of ROI
    get_z_buffer : int, optional
        the buffer of ROI, by default 0
        it is suitable when the given ROI is points rather than polygons. Given this paramter will generate a round buffer
            polygon first, then extract the z-value by this region, but the return will only be a single point
        The unit of buffer follows the ROI coordinates, either pixel or meter.
    shp_proj : str or pyproj.CRS object, optional
        The projection coordinate of given shp file, by default None, it will automatic find the proj file
    geotiff_proj : pyproj.CRS object, optional
        The projection coordinate of given dsm file, by default None, it will automatic find it in geohead
    geo_head : dict, optional
        returned dict of geotiff.get_header() , by default None
        specify this to save the dsm file reading time cost
    name_field : int | str | list, optional
        The column name of shp file to use as index of ROI, by default None
        e.g. shp file with following title:
             | id | plot | species |
             |----|------|---------|
             | 01 | aaa  |   xxx   |
             | 02 | bbb  |   yyy   |

        - "int": the colume number used as title, start from 0
            e.g. name_field = 0 
                -> {"01": [...], "02": [...], ...}
        - "str": the colume title used as title
            e.g. name_field = "plot" 
                -> {"aaa": [...], "bbb": [...]}
        - "list": combine multipule title together
            e.g.: name_field = ["plot", "species"]
                -> {"aaa_xxx": [...], "bbb_yyy":[...], ...}
    title_include : bool, optional
        where add column title into index, by default False
        e.g.: name_field = ["plot", "species"]
            title_include = False
                -> {"aaa_xxx": [...], "bbb_yyy":[...], ...}
            title_include = True
                -> {"plot_aaa_species_xxx": [...], "plot_bbb_species_yyy":[...], ...}
    encoding : str, optional
        by default 'utf-8'

    Returns
    -------
    [type]
        [description]
    """
    
    shp_dict = {}
    if geo_head is None:
        tiff_header = get_header(dsm_path)
    else:
        tiff_header = geo_head

    if geotiff_proj is None:
        shp_dict_2d = read_shp2d(shp_path, geotiff_proj=tiff_header['proj'], shp_proj=shp_proj, 
                                 name_field=name_field, title_include=title_include, encoding=encoding)
    else:
        shp_dict_2d = read_shp2d(shp_path, geotiff_proj=geotiff_proj, shp_proj=shp_proj, 
                                 name_field=name_field, title_include=title_include, encoding=encoding)

    keys = list(shp_dict_2d.keys())
    coord_list = [shp_dict_2d[k] for k in keys]

    if get_z_buffer != 0:
        coord_list_buffer = []
        for k in keys:
            seed = shp_dict_2d[k]
            if seed.shape[0] == 1:   # single point
                buffered = Point(seed[0,:]).buffer(get_z_buffer)
            else:   # polygon
                buffered = Polygon(seed).buffer(get_z_buffer)
            coord_list_buffer.append(np.asarray(buffered.exterior.coords))

    print(f"[io][shp][name] Loading Z values from DSM, this may take a while")
    # then add z_values on it
    if get_z_by == 'local':
        z_lists = point_query(dsm_path, coord_list, geo_head=tiff_header)
        for k, coord_np, coord_z in zip(keys, coord_list, z_lists):
            coord_np = np.concatenate([coord_np, coord_z[:, None]], axis=1)
            shp_dict[k] = coord_np
    elif get_z_by == 'mean':
        if get_z_buffer == 0:
            z_lists = mean_values(dsm_path, polygon=coord_list, geo_head=tiff_header)
        else:
            z_lists = mean_values(dsm_path, polygon=coord_list_buffer, geo_head=tiff_header)

        for k, coord_np, coord_z in zip(keys, coord_list, z_lists):
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)
            shp_dict[k] = coord_np
    elif get_z_by == 'min':
        if get_z_buffer == 0:
            z_lists = min_values(dsm_path, polygon=coord_list, geo_head=tiff_header)
        else:
            z_lists = min_values(dsm_path, polygon=coord_list_buffer, geo_head=tiff_header)

        for k, coord_np, coord_z in zip(keys, coord_list, z_lists):
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)
            shp_dict[k] = coord_np
    elif get_z_by == 'max':
        if get_z_buffer == 0:
            z_lists = min_values(dsm_path, polygon=coord_list, geo_head=tiff_header, pctl=95)
        else:
            z_lists = min_values(dsm_path, polygon=coord_list_buffer, geo_head=tiff_header, pctl=95)
        for k, coord_np, coord_z in zip(keys, coord_list, z_lists):
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)
            shp_dict[k] = coord_np
    elif get_z_by == 'all':
        coord_z = mean_values(dsm_path, polygon='all', geo_head=tiff_header)
        for k, coord_np in zip(keys, coord_list):
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)
            shp_dict[k] = coord_np

    return shp_dict