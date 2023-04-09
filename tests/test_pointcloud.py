import re
import numpy as np
import pytest
import shutil

import easyidp as idp

test_data = idp.data.TestData()
from . import roi_select

##########################
# test read point clouds #
##########################

def test_def_read_ply_binary():
    points, colors, normals = idp.pointcloud.read_ply(test_data.pcd.lotus_ply_bin)

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
    points, colors, normals = idp.pointcloud.read_ply(test_data.pcd.lotus_ply_asc)

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
    points, colors, normals = idp.pointcloud.read_las(test_data.pcd.lotus_las)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 250
    assert colors.min() == 15

def test_def_read_laz():
    points, colors, normals = idp.pointcloud.read_laz(test_data.pcd.lotus_laz)

    # compare results
    assert points.max() == 0.8320407999999999
    assert points.min() == -18.9083118

    assert colors.max() == 250
    assert colors.min() == 15

def test_def_read_las_13ver():
    points, colors, normals = idp.pointcloud.read_las(test_data.pcd.lotus_las13)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 250
    assert colors.min() == 15

def test_def_read_laz_13ver():
    points, colors, normals = idp.pointcloud.read_laz(test_data.pcd.lotus_laz13)

    # compare results
    assert points.max() == 0.8320407
    assert points.min() == -18.9083118

    assert colors.max() == 250
    assert colors.min() == 15

def test_read_ply_with_normals():
    points, colors, normals = idp.pointcloud.read_ply(test_data.pcd.maize_ply)

    assert normals.shape == (49658, 3)

def test_read_las_with_normals():
    points, colors, normals = idp.pointcloud.read_las(test_data.pcd.maize_las)

    assert normals.shape == (49658, 3)

def test_read_laz_with_normals():
    points, colors, normals = idp.pointcloud.read_laz(test_data.pcd.maize_laz)

    assert normals.shape == (49658, 3)

###########################
# test write point clouds #
###########################

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
    out_path_bin  = test_data.pcd.out / "test_def_write_ply_bin.ply"
    out_path_asc  = test_data.pcd.out / "test_def_write_ply_asc.ply"
    # with normals
    out_npath_bin = test_data.pcd.out / "test_def_write_nply_bin.ply"
    out_npath_asc = test_data.pcd.out / "test_def_write_nply_asc.ply"

    # without normals
    if out_path_bin.exists():
        out_path_bin.unlink()
    if out_path_asc.exists():
        out_path_asc.unlink()
    # with normals
    if out_npath_bin.exists():
        out_npath_bin.unlink()
    if out_npath_asc.exists():
        out_npath_asc.unlink()
    
    # without normals
    idp.pointcloud.write_ply(out_path_bin,  write_points, write_colors, binary=True)
    idp.pointcloud.write_ply(out_path_asc,  write_points, write_colors, binary=False)
    # with normals
    idp.pointcloud.write_ply(out_npath_bin, write_points, write_colors, normals=write_normals, binary=True)
    idp.pointcloud.write_ply(out_npath_asc, write_points, write_colors, normals=write_normals, binary=False)

    # test if file created
    assert out_path_bin.exists()
    assert out_path_asc.exists()

    assert out_npath_bin.exists()
    assert out_npath_asc.exists()

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
    out_path_las = test_data.pcd.out / "test_def_write_las.las"
    # with normals
    out_npath_las = test_data.pcd.out /  "test_def_write_nlas.las"
    
    # without normals
    if out_path_las.exists():
        out_path_las.unlink()
    # with normals
    if out_npath_las.exists():
        out_npath_las.unlink()

    # without normals
    idp.pointcloud.write_las(out_path_las,  write_points, write_colors)
    # with normals
    idp.pointcloud.write_las(out_npath_las, write_points, write_colors, normals=write_normals)

    # test if file created
    assert out_path_las.exists()
    assert out_npath_las.exists()

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
    out_path_laz = test_data.pcd.out / "test_def_write_laz.laz"
    # with normals
    out_npath_laz = test_data.pcd.out /  "test_def_write_nlaz.laz"

    # without normals
    if out_path_laz.exists():
        out_path_laz.unlink()
    # with normals
    if out_npath_laz.exists():
        out_npath_laz.unlink()

    # without normals
    idp.pointcloud.write_laz(out_path_laz,  write_points, write_colors)
    # with normals
    idp.pointcloud.write_laz(out_npath_laz, write_points, write_colors, normals=write_normals)

    # test if file created
    assert out_path_laz.exists()
    assert out_npath_laz.exists()

    p, c, n = idp.pointcloud.read_laz(out_path_laz)
    # where will be a precision loss, wait for the feedback of https://github.com/laspy/laspy/issues/222
    np.testing.assert_almost_equal(p, write_points, decimal=5)
    np.testing.assert_almost_equal(c, write_colors)
    assert n is None

    p, c, n = idp.pointcloud.read_laz(out_npath_laz)
    np.testing.assert_almost_equal(p, write_points, decimal=5)
    np.testing.assert_almost_equal(c, write_colors)
    np.testing.assert_almost_equal(n, write_normals)


###########################
# test class point clouds #
###########################

def test_class_pointcloud_init():
    # test the empty object
    pcd = idp.PointCloud()

    assert pcd.points is None
    assert pcd.colors is None
    assert pcd.normals is None
    np.testing.assert_array_almost_equal(
        pcd._offset, np.array([0., 0., 0.,])
    )

    assert pcd.has_points() == False
    assert pcd.has_colors() == False
    assert pcd.has_normals() == False

    assert pcd.shape == (0, 3)

def test_class_pointcloud_init_wrong_path():
    with pytest.warns(UserWarning, match=re.escape("Can not find file")):
        pcd = idp.PointCloud("a/wrong/path.ply")

def test_class_pointcloud_print():
    # short table
    expected_str_s = '      x    y    z  r       g       b           nx      ny      nz\n 0    1    2    3  nodata  nodata  nodata  nodata  nodata  nodata\n 1    4    5    6  nodata  nodata  nodata  nodata  nodata  nodata\n 2    7    8    9  nodata  nodata  nodata  nodata  nodata  nodata'
    pcd = idp.PointCloud()
    pcd.points = np.asarray([[1,2,3], [4,5,6], [7,8,9]])

    assert pcd._btf_print.replace(' ', '') ==  expected_str_s.replace(' ', '')

    # long table
    expected_str_l = '             x        y        z  r    g    b        nx      ny      nz\n    0  -18.908  -15.778   -0.779  123  103  79   nodata  nodata  nodata\n    1  -18.908  -15.777   -0.78   124  104  81   nodata  nodata  nodata\n    2  -18.907  -15.775   -0.802  123  103  80   nodata  nodata  nodata\n  ...  ...      ...      ...      ...  ...  ...     ...     ...     ...\n42451  -15.789  -17.961   -0.847  116  98   80   nodata  nodata  nodata\n42452  -15.789  -17.939   -0.84   113  95   76   nodata  nodata  nodata\n42453  -15.786  -17.937   -0.833  115  97   78   nodata  nodata  nodata'

    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

    assert pcd._btf_print.replace(' ', '') ==  expected_str_l.replace(' ', '')

def test_class_pointcloud_points_set_value():
    pts1 = np.asarray([[1,2,3], [4,5,6]])
    pts2 = np.asarray([[1,2,3], [4,5,6], [7,8,9]])

    # for empty object
    pcd = idp.PointCloud()

    # should able set any points for emtpy pcd
    ## check input type, only ndarray accepable
    with pytest.raises(TypeError, match=re.escape("Only numpy ndarray object are acceptable for setting values")):
        pcd.points = [[1,2,3], [4,5,6]]

    # set ndarray
    pcd.points = pts1
    assert pcd.shape == (2, 3)

    ## change shape directly is not acceptable
    with pytest.raises(IndexError, match=re.escape("The given shape [(3, 3)] does not match current point cloud shape [(2, 3)]")):
        pcd.points = pts2

    ## should change shape first, then change points
    pcd.shape = pts2.shape
    pcd.points = pts2
    
    np.testing.assert_almost_equal(pcd.points, pts2)

def test_class_pointcloud_offset_wrong_type():
    ## input wrong offset
    with pytest.raises(ValueError, match=re.escape("Please give correct 3D coordinate [x, y, z], only 2 was given")):
        pcd = idp.PointCloud(offset=[367900, 3955800])
    with pytest.raises(ValueError, match=re.escape("Please give correct 3D coordinate [x, y, z], only 2 was given")):
        pcd = idp.PointCloud(offset=np.array([367900, 3955800]))

    with pytest.raises(ValueError, match=re.escape("Only [x, y, z] list or np.array([x, y, z]) are acceptable")):
        pcd = idp.PointCloud(offset={"x": 367900, 'y': 3955800, 'z': 0})

def test_class_pointcloud_offset_set_value():
    # check the points value no change after chaning offsets
    pts1 = np.asarray([[1,2,3], [4,5,6]])

    # check default offset 000
    pcd = idp.PointCloud()

    pcd.points = pts1

    np.testing.assert_almost_equal(pcd.points, pts1)
    np.testing.assert_almost_equal(pcd._points, pts1)

    ## change offset value
    pcd.update_offset_value(np.array([1,1,1]))
    
    np.testing.assert_almost_equal(pcd.points, pts1)
    np.testing.assert_almost_equal(pcd._points, np.array([[0,1,2],[3,4,5]]))

    # check create with offset
    pcd = idp.PointCloud(offset=[10,10,10])

    pcd.points = pts1

    np.testing.assert_almost_equal(pcd.points, pts1)
    np.testing.assert_almost_equal(pcd._points, np.array([[-9, -8, -7],[-6, -5, -4]]))

    # then change offset
    pcd.update_offset_value([0, 0, 0])

    np.testing.assert_almost_equal(pcd.points, pts1)
    np.testing.assert_almost_equal(pcd._points, pts1)

def test_class_pointcloud_def_read_point_cloud_no_offset():
    # read pcd without offset
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

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
        pcd._offset, np.array([0., 0., 0.,])
    )

    assert pcd.has_points()
    assert pcd.has_colors()
    assert pcd.has_normals() == False

def test_class_pointcloud_def_read_point_cloud_with_offsets():
    # read pcd with large offset (e.g. xyz in UTM Geo coordinates)
    pcd = idp.PointCloud(test_data.pcd.maize_las)

    np.testing.assert_almost_equal(pcd._offset, np.array([ 367900., 3955800., 0.]))
    np.testing.assert_almost_equal(pcd._points[0,:], np.array([ 93.0206,  65.095 ,  57.9707]))

    # specify origin offset
    ## input as list
    pcd = idp.PointCloud(offset=[367900, 3955800, 0])
    np.testing.assert_almost_equal(pcd._offset, np.array([ 367900., 3955800., 0.]))
    ## input as tuple
    pcd = idp.PointCloud(offset=(367900, 3955800, 0))
    np.testing.assert_almost_equal(pcd._offset, np.array([ 367900., 3955800., 0.]))
    ## input as int array
    pcd = idp.PointCloud(offset=np.array([367900, 3955800, 0]))
    np.testing.assert_almost_equal(pcd._offset, np.array([ 367900., 3955800., 0.]))


def test_class_pointcloud_def_write_point_cloud():
    pcd = idp.PointCloud(test_data.pcd.maize_las)
    
    # test default ext same as input
    expected_file = test_data.pcd.out / "test_class_write_pcd.las"
    if expected_file.exists():
        expected_file.unlink()

    with pytest.warns(UserWarning, match=re.escape("It seems file")):
        save_path = test_data.pcd.out / "test_class_write_pcd"
        pcd.write_point_cloud(save_path)
        assert expected_file.exists()

    # test specify another ext
    expected_file = test_data.pcd.out / "test_class_write_pcd.ply"
    if expected_file.exists():
        expected_file.unlink()

    pcd.write_point_cloud(expected_file)
    assert expected_file.exists()

    # test raise error if format not in 3 specified formats
    with pytest.raises(IOError, match=re.escape("Only support point cloud file format")):
        error_file = test_data.pcd.out / "test_class_write_pcd.pcd"
        pcd.write_point_cloud(error_file)

def test_class_pointcloud_clear():
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

    pcd.clear()

    assert pcd.points is None
    assert pcd.colors is None
    assert pcd.normals is None
    np.testing.assert_array_almost_equal(
        pcd._offset, np.array([0., 0., 0.,])
    )

    assert pcd.has_points() == False
    assert pcd.has_colors() == False
    assert pcd.has_normals() == False

    assert pcd.shape == (0, 3)

def test_class_point_cloud_crop():
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

    polygon = np.array([
        [-18.42576599, -16.10819054],
        [-18.00066757, -18.05295944],
        [-16.05021095, -17.63488388],
        [-16.46848488, -15.66774559],
        [-18.42576599, -16.10819054]])

    cropped = pcd.crop_point_cloud(polygon)

    # check if the type is point cloud
    assert isinstance(cropped, idp.PointCloud)
    # check the point number
    assert cropped.shape[0] == 22422

    # check raise error
    p1 = [[1,2,3], [4,5,6]]
    p2 = np.array(p1)
    with pytest.raises(TypeError, match=re.escape(
        "Only numpy ndarray are supported as `polygon_xy` inputs, not <class 'list'>")):
        cropped = pcd.crop_point_cloud(p1)

    with pytest.raises(IndexError, match=re.escape(
        "Please only spcify shape like (N, 2), not (2, 3)")):
        cropped = pcd.crop_point_cloud(p2)

    # check raise warns
    with pytest.warns(UserWarning, match=re.escape(
        "Cropped 0 point in given polygon. Please check whether the coords is correct.")):
        cropped = pcd.crop_point_cloud(polygon + 10)
        assert cropped is None

def test_class_crop():
    roi = roi_select.copy()
    roi.get_z_from_dsm(test_data.pix4d.lotus_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)

    p4d = idp.Pix4D(
        project_path=test_data.pix4d.lotus_folder,
        param_folder=test_data.pix4d.lotus_param
    )
    p4d.load_pcd(test_data.pix4d.lotus_pcd)

    tif_out_folder = test_data.pcd.out / "class_crop"
    # clear temp output folder
    if tif_out_folder.exists():
        shutil.rmtree(tif_out_folder)
    tif_out_folder.mkdir()

    out = p4d.pcd.crop_rois(roi, save_folder=tif_out_folder)

    assert len(out) == 4
    assert len(out["N1W1"]) == 15226

    assert (tif_out_folder / "N1W1.ply").exists()

    # also need check the offsets and points values
    # should belong to -> test_class_point_cloud_crop()
    # but previous function have no pix4d offset
    np.testing.assert_almost_equal(out["N1W1"].offset, p4d.pcd.offset)
    assert np.all(out["N1W1"]._points[:, 0:2] < 300)