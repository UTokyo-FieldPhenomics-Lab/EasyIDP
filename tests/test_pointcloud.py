from asyncore import write
import os
import sys
import easyidp as idp
import numpy as np

##########################
# test read point clouds #
##########################

data_path =  "./tests/data/pcd_test"

def test_def_read_ply_binary():
    ply_binary_path = os.path.join(data_path, "hasu_tanashi_binary.ply")

    points, colors = idp.pointcloud.read_ply(ply_binary_path)

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

    points, colors = idp.pointcloud.read_ply(ply_ascii_path)

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

def test_def_read_laz_las():
    las_path = os.path.join(data_path, "hasu_tanashi.las")

    points, colors = idp.pointcloud.read_laz(las_path)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_laz_laz():
    laz_path = os.path.join(data_path, "hasu_tanashi.laz")

    points, colors = idp.pointcloud.read_laz(laz_path)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_laz_las13ver():
    las_path = os.path.join(data_path, "hasu_tanashi_1.3.las")

    points, colors = idp.pointcloud.read_laz(las_path)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0

def test_def_read_laz_laz13ver():
    laz_path = os.path.join(data_path, "hasu_tanashi_1.3.laz")

    points, colors = idp.pointcloud.read_laz(laz_path)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 224
    assert colors.min() == 0


###########################
# test write point clouds #
###########################

out_path = "./tests/out/pcd_test"

write_points = np.asarray([[-18.9083118, -15.7775583,  -0.77878  ],
                           [-18.9082794, -15.7772741,  -0.7802601],
                           [-18.907196 , -15.7748289,  -0.8017483],
                           [-15.7892904, -17.9612598,  -0.8468666],
                           [-15.7885809, -17.9391041,  -0.839632 ],
                           [-15.7862186, -17.9365788,  -0.8327141]])
write_colors = np.asarray([[  0,   0,   0],
                           [  0,   0,   0],
                           [  0,   0,   0],
                           [192,  64, 128],
                           [ 92,  88,  83],
                           [ 64,  64,  64]])


def test_def_write_ply():
    out_path_bin = os.path.join(out_path, "test_def_write_ply_bin.ply")
    out_path_asc = os.path.join(out_path, "test_def_write_ply_asc.ply")

    if os.path.exists(out_path_bin):
        os.remove(out_path_bin)
    if os.path.exists(out_path_asc):
        os.remove(out_path_asc)

    idp.pointcloud.write_ply(write_points, write_colors, out_path_bin, binary=True)
    idp.pointcloud.write_ply(write_points, write_colors, out_path_asc, binary=False)

    assert os.path.exists(out_path_bin)
    assert os.path.exists(out_path_asc)


def test_def_write_las():
    out_path_las = os.path.join(out_path, "test_def_write_las.las")
    out_path_laz = os.path.join(out_path, "test_def_write_las.laz")

    if os.path.exists(out_path_las):
        os.remove(out_path_las)
    if os.path.exists(out_path_laz):
        os.remove(out_path_laz)

    idp.pointcloud.write_laz(write_points, write_colors, out_path_laz)
    idp.pointcloud.write_las(write_points, write_colors, out_path_las)

    assert os.path.exists(out_path_las)
    assert os.path.exists(out_path_laz)


###########################
# test class point clouds #
###########################

def test_class_pointcloud_init():
    pcd = idp.pointcloud()