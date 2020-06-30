import numpy as np
import pyproj
from easyric.external import shapefile
from easyric.io.geotiff import point_query, mean_values

def read_proj(prj_path):
    with open(prj_path, 'r') as f:
        wkt_string = f.readline()
    proj = pyproj.CRS.from_wkt(wkt_string)

    return proj

def read_shp2d(shp_path, target_proj=None):
    shp = shapefile.Reader(shp_path)
    shp_dict = {}

    for i, shape in enumerate(shp.shapes()):
        plot_name = shp.records()[i][-1]
        if isinstance(plot_name, str):
            plot_name = plot_name.replace(r'/', '_')
            plot_name = plot_name.replace(r'\\', '_')
        else:
            plot_name = str(plot_name)

        coord_np = np.asarray(shape.points)
        shp_dict[plot_name] = coord_np

    return shp_dict

def read_shp3d(shp_path, dsm_path, get_z_by='mean', target_proj=None):
    '''
    shp_path: full shp_file directory
    get_z_by:
        "mean": using the mean value of whole DSM as the same z-value for all boundary points
                -> this will get a 2D plane of ROI
        "local" using the z value of where boundary points located, each point will get different z-values
                -> this will get a 3D curved mesh of ROI
    '''
    shp = shapefile.Reader(shp_path)
    shp_dict = {}

    for i, shape in enumerate(shp.shapes()):
        plot_name = shp.records()[i][-1]
        if isinstance(plot_name, str):
            plot_name = plot_name.replace(r'/', '_')
            plot_name = plot_name.replace(r'\\', '_')
        else:
            plot_name = str(plot_name)
        coord_np = np.asarray(shape.points)

        if get_z_by == 'local':
            coord_z = point_query(dsm_path, coord_np)
            coord_np = np.concatenate([coord_np, coord_z[:, None]], axis=1)
        elif get_z_by == 'mean':
            coord_z = mean_values(dsm_path)
            coord_np = np.insert(coord_np, obj=2, values=coord_z, axis=1)

        shp_dict[plot_name] = coord_np

    return shp_dict