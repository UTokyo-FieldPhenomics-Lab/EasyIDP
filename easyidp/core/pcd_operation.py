import numpy as np
import open3d as o3d


def merge_pcd(o3d_pcd_list):
    final_pcd = o3d.geometry.PointCloud()
    xyz = np.empty((0, 3))
    rgb = np.empty((0, 3))

    for pcd in o3d_pcd_list:
        pcd_xyz = np.asarray(pcd.points)
        pcd_rgb = np.asarray(pcd.colors)

        xyz = np.vstack([xyz, pcd_xyz])
        rgb = np.vstack([rgb, pcd_rgb])

    final_pcd.points = o3d.utility.Vector3dVector(xyz)
    final_pcd.colors = o3d.utility.Vector3dVector(rgb)

    return final_pcd


def crop_pcd_xy(o3d_pcd, polygon, z_range=None):
    """
    Parameters
    ----------
    o3d_pcd:

    polygon: np.ndarray 
        shape=[n x 3]

    z_range: tuple or None

    """
    if z_range is None:
        points_np = np.asarray(o3d_pcd.points)
        _, _, zmin = points_np.min(axis=0)
        _, _, zmax = points_np.max(axis=0)
    else:
        zmin = z_range[0]
        zmax = z_range[1]

    o3d_boundary = _build_xy_crop_boundary(polygon, (zmin, zmax))

    return o3d_boundary.crop_point_cloud(o3d_pcd)


def _build_xy_crop_boundary(polygon, z_range=(-100, 100)):
    """
    Parameters
    ----------
    polygon: np.ndarray 
        shape=[n x 3]

    z_range: list or tuple
        z_range=(z_min, z_max)

    Returns
    -------
    o3d_boundary: open3d.visualization.SelectionPolygonVolume()
    """
    z_min = z_range[0]
    z_max = z_range[1]

    o3d_boundary = o3d.visualization.SelectionPolygonVolume()
    o3d_boundary.orthogonal_axis = "Z"
    o3d_boundary.bounding_polygon = o3d.utility.Vector3dVector(polygon)
    o3d_boundary.axis_max = z_max
    o3d_boundary.axis_min = z_min

    return o3d_boundary


