from asyncore import write
import os
import re
import sys
import easyidp as idp
import numpy as np
import pytest

##########################
# test read point clouds #
##########################

data_path =  "./tests/data/pcd_test"

def test_def_read_ply_binary():
    ply_binary_path = os.path.join(data_path, "hasu_tanashi_binary.ply")

    points, colors, normals = idp.pointcloud.read_ply(ply_binary_path)

    # comare results
    np.testing.assert_array_almost_equal(
        points[ 0, :], 
        np.array([-18.908312, -15.777558,  -0.77878 ], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        points[-1, :],
        np.array([-15.786219 , -17.936579 ,  -0.8327141], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        colors[ 0, :], 
        np.array([123, 103,  79], dtype=np.uint8)
    )
    np.testing.assert_array_almost_equal(
        colors[-1, :],
        np.array([115,  97,  78], dtype=np.uint8)
    )

def test_def_read_ply_ascii():
    ply_ascii_path = os.path.join(data_path, "hasu_tanashi_ascii.ply")

    points, colors, normals = idp.pointcloud.read_ply(ply_ascii_path)

    # compare results
    np.testing.assert_array_almost_equal(
        points[ 0, :], 
        np.array([-18.908312, -15.777558,  -0.77878 ], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        points[-1, :],
        np.array([-15.786219 , -17.936579 ,  -0.8327141], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        colors[ 0, :], 
        np.array([123, 103,  79], dtype=np.uint8)
    )
    np.testing.assert_array_almost_equal(
        colors[-1, :],
        np.array([115,  97,  78], dtype=np.uint8)
    )

def test_def_read_las():
    las_path = os.path.join(data_path, "hasu_tanashi.las")

    points, colors, normals = idp.pointcloud.read_las(las_path)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_laz():
    laz_path = os.path.join(data_path, "hasu_tanashi.laz")

    points, colors, normals = idp.pointcloud.read_laz(laz_path)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_las_13ver():
    las_path = os.path.join(data_path, "hasu_tanashi_1.3.las")

    points, colors, normals = idp.pointcloud.read_las(las_path)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_laz_13ver():
    laz_path = os.path.join(data_path, "hasu_tanashi_1.3.laz")

    points, colors, normals = idp.pointcloud.read_laz(laz_path)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_read_ply_with_normals():
    ply_path = os.path.join(data_path, "maize3na_20210614_15m_utm.ply")

    points, colors, normals = idp.pointcloud.read_ply(ply_path)

    assert normals.shape == (49658, 3)

def test_read_las_with_normals():
    las_path = os.path.join(data_path, "maize3na_20210614_15m_utm.las")
    points, colors, normals = idp.pointcloud.read_las(las_path)

    assert normals.shape == (49658, 3)

def test_read_laz_with_normals():
    laz_path = os.path.join(data_path, "maize3na_20210614_15m_utm.laz")
    points, colors, normals = idp.pointcloud.read_laz(laz_path)

    assert normals.shape == (49658, 3)

###########################
# test write point clouds #
###########################

out_path = "./tests/out/pcd_test"

write_points = np.asarray([[-1.9083118, -1.7775583,  -0.77878  ],
                           [-1.9082794, -1.7772741,  -0.7802601],
                           [-1.907196 , -1.7748289,  -0.8017483],
                           [-1.7892904, -1.9612598,  -0.8468666],
                           [-1.7885809, -1.9391041,  -0.839632 ],
                           [-1.7862186, -1.9365788,  -0.8327141]], dtype=np.float64)
write_colors = np.asarray([[  0,   0,   0],
                           [  0,   0,   0],
                           [  0,   0,   0],
                           [192,  64, 128],
                           [ 92,  88,  83],
                           [ 64,  64,  64]], dtype=np.uint8)
write_normals = np.asarray([[-0.03287353,  0.36604664,  0.9300157 ],
                            [ 0.08860216,  0.07439037,  0.9932853 ],
                            [-0.01135951,  0.2693031 ,  0.9629885 ],
                            [ 0.4548034 , -0.15576138,  0.876865  ],
                            [ 0.4550802 , -0.29450312,  0.8403392 ],
                            [ 0.32758632,  0.27255052,  0.9046565 ]], dtype=np.float64)

def test_def_write_ply():
    # without normals
    out_path_bin = os.path.join(out_path, "test_def_write_ply_bin.ply")
    out_path_asc = os.path.join(out_path, "test_def_write_ply_asc.ply")
    # with normals
    out_npath_bin = os.path.join(out_path, "test_def_write_nply_bin.ply")
    out_npath_asc = os.path.join(out_path, "test_def_write_nply_asc.ply")

    # without normals
    if os.path.exists(out_path_bin):
        os.remove(out_path_bin)
    if os.path.exists(out_path_asc):
        os.remove(out_path_asc)
    # with normals
    if os.path.exists(out_npath_bin):
        os.remove(out_npath_bin)
    if os.path.exists(out_npath_asc):
        os.remove(out_npath_asc)
    
    # without normals
    idp.pointcloud.write_ply(write_points, write_colors, out_path_bin, binary=True)
    idp.pointcloud.write_ply(write_points, write_colors, out_path_asc, binary=False)
    # with normals
    idp.pointcloud.write_ply(write_points, write_colors, out_npath_bin, normals=write_normals, binary=True)
    idp.pointcloud.write_ply(write_points, write_colors, out_npath_asc, normals=write_normals, binary=False)

    # test if file created
    assert os.path.exists(out_path_bin)
    assert os.path.exists(out_path_asc)

    assert os.path.exists(out_npath_bin)
    assert os.path.exists(out_npath_asc)

    # test value if same
    p, c, n = idp.pointcloud.read_ply(out_path_bin)
    np.testing.assert_almost_equal(p, write_points)
    np.testing.assert_almost_equal(c, write_colors)
    assert n is None

    p, c, n = idp.pointcloud.read_ply(out_path_asc)
    np.testing.assert_almost_equal(p, write_points)
    np.testing.assert_almost_equal(c, write_colors)
    assert n is None

    p, c, n = idp.pointcloud.read_ply(out_npath_bin)
    np.testing.assert_almost_equal(p, write_points)
    np.testing.assert_almost_equal(c, write_colors)
    np.testing.assert_almost_equal(n, write_normals)

    p, c, n = idp.pointcloud.read_ply(out_npath_asc)
    np.testing.assert_almost_equal(p, write_points)
    np.testing.assert_almost_equal(c, write_colors)
    np.testing.assert_almost_equal(n, write_normals)


def test_def_write_las():
    # without normals
    out_path_las = os.path.join(out_path, "test_def_write_las.las")
    # with normals
    out_npath_las = os.path.join(out_path, "test_def_write_nlas.las")
    
    # without normals
    if os.path.exists(out_path_las):
        os.remove(out_path_las)
    # with normals
    if os.path.exists(out_npath_las):
        os.remove(out_npath_las)

    # without normals
    idp.pointcloud.write_las(write_points, write_colors, out_path_las)
    # with normals
    idp.pointcloud.write_las(write_points, write_colors, out_npath_las, normals=write_normals)

    # test if file created
    assert os.path.exists(out_path_las)
    assert os.path.exists(out_npath_las)

    p, c, n = idp.pointcloud.read_las(out_path_las)
    # where will be a precision loss, wait for the feedback of https://github.com/laspy/laspy/issues/222
    np.testing.assert_almost_equal(p, write_points, decimal=2)
    np.testing.assert_almost_equal(c, write_colors)
    assert n is None

    p, c, n = idp.pointcloud.read_las(out_npath_las)
    np.testing.assert_almost_equal(p, write_points, decimal=2)
    np.testing.assert_almost_equal(c, write_colors)
    np.testing.assert_almost_equal(n, write_normals)


def test_def_write_laz():
    # without normals
    out_path_laz = os.path.join(out_path, "test_def_write_laz.laz")
    # with normals
    out_npath_laz = os.path.join(out_path, "test_def_write_nlaz.laz")

    # without normals
    if os.path.exists(out_path_laz):
        os.remove(out_path_laz)
    # with normals
    if os.path.exists(out_npath_laz):
        os.remove(out_npath_laz)

    # without normals
    idp.pointcloud.write_laz(write_points, write_colors, out_path_laz)
    # with normals
    idp.pointcloud.write_laz(write_points, write_colors, out_npath_laz, normals=write_normals)

    # test if file created
    assert os.path.exists(out_path_laz)
    assert os.path.exists(out_npath_laz)

    p, c, n = idp.pointcloud.read_laz(out_path_laz)
    # where will be a precision loss, wait for the feedback of https://github.com/laspy/laspy/issues/222
    np.testing.assert_almost_equal(p, write_points, decimal=2)
    np.testing.assert_almost_equal(c, write_colors)
    assert n is None

    p, c, n = idp.pointcloud.read_laz(out_npath_laz)
    np.testing.assert_almost_equal(p, write_points, decimal=2)
    np.testing.assert_almost_equal(c, write_colors)
    np.testing.assert_almost_equal(n, write_normals)


###########################
# test class point clouds #
###########################

def test_class_pointcloud_init():
    # test the empty project
    pcd = idp.PointCloud()

    assert pcd.points is None
    assert pcd.colors is None
    assert pcd.normals is None
    np.testing.assert_array_almost_equal(
        pcd.offset, np.array([0., 0., 0.,])
    )

    assert pcd.has_points() == False
    assert pcd.has_colors() == False
    assert pcd.has_normals() == False

def test_class_pointcloud_def_read_point_cloud_no_offset():
    # read pcd without offset
    pcd = idp.PointCloud(os.path.join(data_path, "hasu_tanashi_binary.ply"))

    assert pcd.points is not None
    np.testing.assert_array_almost_equal(
        pcd.points[ 0, :], 
        np.array([-18.908312, -15.777558,  -0.77878 ], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        pcd.points[-1, :],
        np.array([-15.786219 , -17.936579 ,  -0.8327141], dtype=np.float32)
    )

    assert pcd.colors is not None
    np.testing.assert_array_almost_equal(
        pcd.colors[ 0, :], 
        np.array([123, 103,  79], dtype=np.uint8)
    )
    np.testing.assert_array_almost_equal(
        pcd.colors[-1, :],
        np.array([115,  97,  78], dtype=np.uint8)
    )

    assert pcd.normals is None

    np.testing.assert_array_almost_equal(
        pcd.offset, np.array([0., 0., 0.,])
    )

    assert pcd.has_points()
    assert pcd.has_colors()
    assert pcd.has_normals() == False


def test_class_pointcloud_def_read_point_cloud_offsets():
    # read pcd with large offset (e.g. xyz in UTM Geo coordinates)
    pcd = idp.PointCloud(os.path.join(data_path, "maize3na_20210614_15m_utm.las"))

    np.testing.assert_almost_equal(pcd.offset, np.array([ 367900., 3955800., 0.]))
    np.testing.assert_almost_equal(pcd.points[0,:], np.array([ 93.0206,  65.095 ,  57.9707]))

    # specify origin offset
    ## input as list
    pcd = idp.PointCloud(offset=[367900, 3955800, 0])
    np.testing.assert_almost_equal(pcd.offset, np.array([ 367900., 3955800., 0.]))
    ## input as tuple
    pcd = idp.PointCloud(offset=(367900, 3955800, 0))
    np.testing.assert_almost_equal(pcd.offset, np.array([ 367900., 3955800., 0.]))
    ## input as int array
    pcd = idp.PointCloud(offset=np.array([367900, 3955800, 0]))
    np.testing.assert_almost_equal(pcd.offset, np.array([ 367900., 3955800., 0.]))

    ## input wrong offset
    with pytest.raises(ValueError, match=re.escape("Please give correct 3D coordinate [x, y, z], only 2 was given")):
        pcd = idp.PointCloud(offset=[367900, 3955800])
    with pytest.raises(ValueError, match=re.escape("Please give correct 3D coordinate [x, y, z], only 2 was given")):
        pcd = idp.PointCloud(offset=np.array([367900, 3955800]))

    with pytest.raises(ValueError, match=re.escape("Only [x, y, z] list or np.array([x, y, z]) are acceptable")):
        pcd = idp.PointCloud(offset={"x": 367900, 'y': 3955800, 'z': 0})


def test_class_pointcloud_def_write_point_cloud():
    pcd = idp.PointCloud(os.path.join(data_path, "maize3na_20210614_15m_utm.las"))
    

    # test default ext same as input
    expected_file = os.path.join(out_path, "test_class_write_pcd.las")
    if os.path.exists(expected_file):
        os.remove(expected_file)

    with pytest.warns(UserWarning, match=re.escape("It seems file")):
        save_path = os.path.join(out_path, "test_class_write_pcd")
        pcd.write_point_cloud(save_path)
        assert os.path.exists(expected_file)

    # test specify another ext
    expected_file = os.path.join(out_path, "test_class_write_pcd.ply")
    if os.path.exists(expected_file):
        os.remove(expected_file)

    pcd.write_point_cloud(expected_file)
    assert os.path.exists(expected_file)

    # test raise error if format not in 3 specified formats
    with pytest.raises(IOError, match=re.escape("Only support point cloud file format")):
        error_file = os.path.join(out_path, "test_class_write_pcd.pcd")
        pcd.write_point_cloud(error_file)