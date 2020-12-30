import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from skimage.morphology import disk
from skimage import filters

def calculate_xyz_volume(pcd):
    pcd_xyz = np.asarray(pcd.points)

    x_len = pcd_xyz[:, 0].max() - pcd_xyz[:, 0].min()
    y_len = pcd_xyz[:, 1].max() - pcd_xyz[:, 1].min()
    z_len = pcd_xyz[:, 2].max() - pcd_xyz[:, 2].min()

    return x_len * y_len * z_len

def merge_pcd(pcd_list):
    final_pcd = o3d.geometry.PointCloud()
    xyz = np.empty((0, 3))
    rgb = np.empty((0, 3))

    for pcd in pcd_list:
        pcd_xyz = np.asarray(pcd.points)
        pcd_rgb = np.asarray(pcd.colors)

        xyz = np.vstack([xyz, pcd_xyz])
        rgb = np.vstack([rgb, pcd_rgb])

    final_pcd.points = o3d.utility.Vector3dVector(xyz)
    final_pcd.colors = o3d.utility.Vector3dVector(rgb)

    return final_pcd

def build_cut_boundary(polygon, z_range):
    """
    :param polygon: np.array shape=[n x 3]
    :param z_range: list or tuple, z_range=(z_min, z_max)
    """
    z_min = z_range[0]
    z_max = z_range[1]

    boundary = o3d.visualization.SelectionPolygonVolume()
    boundary.orthogonal_axis = "Z"
    boundary.bounding_polygon = o3d.utility.Vector3dVector(polygon)
    boundary.axis_max = z_max
    boundary.axis_min = z_min

    return boundary

def clip_pcd(pcd, boundary):
    pass

def get_convex_hull(pcd, dim='2d'):
    # in scipy, 2D hull.area is perimeter, hull.volume is area
    # https://stackoverflow.com/questions/35664675/in-scipys-convexhull-what-does-area-measure
    #
    # >>> points = np.array([[-1,-1], [1,1], [-1, 1], [1,-1]])
    # >>> hull = ConvexHull(points)
    # ==== 2D ====
    # >>> print(hull.volume)
    # 4.00
    # >>> print(hull.area)
    # 8.00
    # ==== 3D ====
    # >>> points = np.array([[-1,-1, -1], [-1,-1, 1],
    # ...                    [-1, 1, -1], [-1, 1, 1],
    # ...                    [1, -1, -1], [1, -1, 1],
    # ...                    [1,  1, -1], [1,  1, 1]])
    # >>> hull = ConvexHull(points)
    # >>> hull.area
    # 24.0
    # >>> hull.volume
    # 8.0
    pcd_xyz = np.asarray(pcd.points)
    if dim == '2d' or dim == '2D':
        xy = pcd_xyz[:, 0:2]
        hull = ConvexHull(xy)
    elif dim == '3d' or dim == '3D':
        hull = ConvexHull(xyz)
    else:
        raise KeyError('Only "2d" and "3d" or "2D" and "3D" are acceptable for dim parameters')
    hull_volume = hull.volume
    hull_xy = xy[hull.vertices, :]
    return hull_xy, hull_volume

def round2val(a, round_val):
    return np.floor( np.array(a, dtype=float) / round_val) * round_val

def pcd2dxm(pcd, dens=1, interp=True):
    # dens = how many points per pixel, default is 1 (highest resolution)
    rua = pd.DataFrame(np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)*255]), columns=['x','y','z','r', 'g', 'b'])
    rua_len = rua.max() - rua.min()
    
    grid_num = int(np.ceil(len(rua) / dens))
    # x/res * y/res = grid_num
    # x * y / res^2 = grid_num
    # x * y / grid_num = res^2
    res = np.sqrt(rua_len['x'] * rua_len['y'] / grid_num)
    
    x_num = int(np.ceil(rua_len['x'] / res))
    y_num = int(np.ceil(rua_len['y'] / res))
    
    rua['x_round'] = round2val(rua['x'], res)
    rua['y_round'] = round2val(rua['y'], res)
    
    rua['x_pos'] = rua['x_round'] / res
    rua['y_pos'] = rua['y_round'] / res
    x_pos_min = rua['x_pos'].min()
    y_pos_min = rua['y_pos'].min()
    rua['x_pos'] = rua['x_pos'] - x_pos_min
    rua['y_pos'] = rua['y_pos'] - y_pos_min
    
    group_xy = rua.groupby(['x_round', 'y_round'])
    group_mean = group_xy['z'].max()
    idx = group_xy['z'].transform(max) == rua['z']
    rua_grid = rua[idx]
    
    dsm = np.ones((x_num+1, y_num+1)) * np.nan
    dsm[rua['x_pos'].astype(int), rua['y_pos'].astype(int)] = rua['z']
    
    dom = np.zeros((x_num+1, y_num+1, 4))
    dom[rua['x_pos'].astype(int), rua['y_pos'].astype(int), 0] = rua['r']
    dom[rua['x_pos'].astype(int), rua['y_pos'].astype(int), 1] = rua['g']
    dom[rua['x_pos'].astype(int), rua['y_pos'].astype(int), 2] = rua['b']
    dom[rua['x_pos'].astype(int), rua['y_pos'].astype(int), 3] = 255
    dom = dom.astype(np.uint8)
    
    del rua, rua_grid, group_xy, group_mean
    
    if interp:
        # find the boundary of point clouds
        plane_hull, _ = get_convex_hull(pcd, dim='2d')
        plane_hull = np.vstack([plane_hull, plane_hull[0, :]])
        plane_hull = round2val(plane_hull, res) / res
        plane_hull = np.int_(plane_hull)

        plane_hull[:, 0] = plane_hull[:, 0] - int(x_pos_min)
        plane_hull[:, 1] = plane_hull[:, 1] - int(y_pos_min)

        # find all the empty pixels
        holes = np.argwhere(dom[:,:,3]==0)
        
        # find all the empty pixels in the boundary to aviod unncessary calculation
        path = Path(plane_hull)
        grid = path.contains_points(holes)
        holes_in = holes[grid]
        # and its mask
        mask = np.zeros((x_num+1, y_num+1))
        mask[holes_in[:,0], holes_in[:,1]] = 1
        mask = mask==1
        
        # interp the DOM
        r = dom[:,:,0]
        g = dom[:,:,1]
        b = dom[:,:,2]

        r_filter = filters.rank.mean(r, selem=disk(7))#, mask=mask)
        g_filter = filters.rank.mean(g, selem=disk(7))#, mask=mask)
        b_filter = filters.rank.mean(b, selem=disk(7))#, mask=mask)

        r[mask] = r_filter[mask]
        g[mask] = g_filter[mask]
        b[mask] = b_filter[mask]
        
        del r_filter, g_filter, b_filter
        a = dom[:,:,3].copy()
        a[mask] = 255
        dom_new = np.stack([r,g,b,a], axis=2)

        # interp the DSM
        #[To be continued, because]
        # Filter.rank.mean only support 
        # > ValueError: Images of type float must be between -1 and 1.

        dom_new = dom_new.astype(np.uint8)
        
        return dom_new, dsm
    else:
        return dom, dsm

def pcd2binary(pcd, dpi=10):
    # dpi suggest < 20
    pcd_xyz = np.asarray(pcd.points)
    # !!!! notice !!!!
    # in numpy image system, Y axis is 0, X axis is 1
    y = pcd_xyz[:, 0]
    x = pcd_xyz[:, 1]

    x_length_m = x.max() - x.min()
    y_length_m = y.max() - y.min()
    px_num_per_cm = int(dpi / 2.54)
    width = int(np.ceil(x_length_m * 100 * px_num_per_cm))
    height = int(np.ceil(y_length_m * 100 * px_num_per_cm))
    ref_x = (x - x.min()) / x_length_m * width
    ref_y = (y - y.min()) / y_length_m * height
    ref_pos = np.vstack([ref_x, ref_y]).T.astype(int)
    ref_pos_rm_dup = np.unique(ref_pos, axis=0)

    out_img = np.zeros((width + 1, height + 1))
    out_img[ref_pos_rm_dup[:, 0], ref_pos_rm_dup[:, 1]] = 1
    out_img = out_img.astype(int)

    left_top_corner = (y.min(), x.min())

    return out_img, px_num_per_cm, left_top_corner

def pcd2voxel(pcd, part=100, voxel_size=None):
    pcd_xyz = np.asarray(pcd.points)
    points_num = pcd_xyz.shape[0]  # get the size of this plot
    if voxel_size is None:
        x_max = pcd_xyz[:, 0].max()
        x_min = pcd_xyz[:, 0].min()
        x_len = x_max - x_min

        y_max = pcd_xyz[:, 1].max()
        y_min = pcd_xyz[:, 1].min()
        y_len = y_max - y_min

        z_max = pcd_xyz[:, 2].max()
        z_min = pcd_xyz[:, 2].min()
        z_len = z_max - z_min

        # param part: how many part of the shortest axis will be split?
        vs = min(x_len, y_len, z_len) / part  # Voxel Size (VS)
    else:
        vs = voxel_size
    # convert point cloud to voxel
    # !! Doesn't work in Open3D 0.9.0.0 !!
    # > pcd_voxel = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd, voxel_size=vs)
    # > voxel_num = len(pcd_voxel.voxels)
    pcd_voxel = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd, voxel_size=vs)

    pcd_vx = pcd.voxel_down_sample(voxel_size=vs) # 63012
    voxel_num = np.asarray(pcd_vx.points).shape[0]
    voxel_density = points_num / voxel_num

    voxel_params = {'voxel_size':vs, 'voxel_density': voxel_density, 'voxel_number': voxel_num}

    return pcd_voxel, voxel_params