import os
import pyproj
import numpy as np
from easyric.external import shapefile
from easyric.io.geotiff import point_query, mean_values, get_header


def read_proj(prj_path):
    with open(prj_path, 'r') as f:
        wkt_string = f.readline()

    proj = pyproj.CRS.from_wkt(wkt_string)

    return proj


def read_shp2d(shp_path, target_proj=None, given_proj=None):
    shp = shapefile.Reader(shp_path)
    shp_dict = {}
    current_proj = None

    # try to find current projection
    if target_proj is not None:
        if given_proj is not None:
            current_proj = given_proj
        else:
            prj_path = shp_path[:-4] + '.prj'
            if os.path.exists(prj_path):
                current_proj = read_proj(shp_path[:-4] + '.prj')
                print(f'[io][shp][proj] find ESRI projection file {prj_path}, and successfully obtain projection '
                      f'{current_proj.coordinate_system}')
            else:
                target_proj = None
                print(f'[io][shp][proj] could not find ESRI projection file {prj_path}, could not operate auto-convention, '
                      f'Please convert projection system manually.')

    for i, shape in enumerate(shp.shapes()):
        plot_name = shp.records()[i][-1]
        if isinstance(plot_name, str):
            plot_name = plot_name.replace(r'/', '_')
            plot_name = plot_name.replace(r'\\', '_')
        else:
            plot_name = str(plot_name)

        coord_np = np.asarray(shape.points)

        if target_proj is not None:
            transformer = pyproj.Transformer.from_proj(current_proj, target_proj)
            # the pyshp package load seems get (lon, lat), however, the pyproj use (lat, lon), so need to revert
            transformed = transformer.transform(coord_np[:,1], coord_np[:,0])
            coord_np = np.asarray(transformed).T

            if True in np.isinf(coord_np):
                raise ValueError(f'Fail to convert points from "{current_proj.name}"(shp projection) to '
                                 f'"{target_proj.name}"(dsm projection), '
                                 f'this may caused by the uncertainty of .prj file strings, please check the coordinate '
                                 f'manually via QGIS Layer Infomation, get the EPGS code, and specify the function argument'
                                 f'read_shp2d(..., given_proj=pyproj.CRS.from_epsg(xxxx))')

        shp_dict[plot_name] = coord_np

    return shp_dict


def read_shp3d(shp_path, dsm_path, get_z_by='mean', given_proj=None):
    '''
    shp_path: full shp_file directory
    get_z_by:
        "mean": using the mean value of whole DSM as the same z-value for all boundary points
                -> this will get a 2D plane of ROI
        "local" using the z value of where boundary points located, each point will get different z-values
                -> this will get a 3D curved mesh of ROI
    '''
    shp_dict = {}
    if get_z_by == 'local':
        tiff_header = get_header(dsm_path)
        shp_dict_2d = read_shp2d(shp_path, target_proj=tiff_header['proj'], given_proj=given_proj)
    else:
        shp_dict_2d = read_shp2d(shp_path)

    keys = list(shp_dict_2d.keys())
    coord_list = [shp_dict_2d[k] for k in keys]

    # then add z_values on it
    if get_z_by == 'local':
        z_lists = point_query(dsm_path, coord_list)
        for k, coord_np, coord_z in zip(keys, coord_list, z_lists):
            coord_np = np.concatenate([coord_np, coord_z[:, None]], axis=1)
            shp_dict[k] = coord_np
    elif get_z_by == 'mean':
        coord_z = mean_values(dsm_path)
        for k, coord_np in zip(keys, coord_list):
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)
            shp_dict[k] = coord_np

    return shp_dict