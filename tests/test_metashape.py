import os
import re
import pytest
import pyproj
import numpy as np
import easyidp as idp

ms_path = "tests/data/metashape"

#########################
# test math calculation #
#########################
def test_apply_transform_matrix():
    matrix = np.asarray([
        [-0.86573098, -0.01489186,  0.08977677,  7.65034123],
        [ 0.06972335,  0.44334391,  0.74589315,  1.85910928],
        [-0.05848325,  0.74899678, -0.43972184, -0.1835615],
        [ 0.,          0.,          0.,          1.]], dtype=np.float)
    matrix2 = np.linalg.inv(matrix)

    point1 = np.array([0.5, 1, 1.5], dtype=np.float)
    out1 = idp.metashape.apply_transform_matrix(point1, matrix2)

    ans1 = np.array([7.96006409,  1.30195288, -2.66971818])
    np.testing.assert_array_almost_equal(out1, ans1, decimal=6)

    np_array = np.array([
        [0.5, 1, 1.5],
        [0.5, 1, 1.5]], dtype=np.float)
    out2 = idp.metashape.apply_transform_matrix(np_array, matrix2)
    ans2 = np.array([
        [7.96006409, 1.30195288, -2.66971818],
        [7.96006409, 1.30195288, -2.66971818]])
    np.testing.assert_array_almost_equal(out2, ans2, decimal=6)


def test_convert_proj3d():
    # transform between geocentric and geodetic
    geocentric = np.array([-3943658.7087006606, 3363404.124223561, 3704651.3067566575])

    # columns=['lon', 'lat', 'alt']
    geodetic = np.array([139.54033578028609, 35.73756358928734, 96.87827569602781])

    geo_d = pyproj.CRS.from_epsg(4326)
    geo_c = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})

    out_c = idp.metashape.convert_proj3d(
        geodetic, geo_d, geo_c, is_geo=True
    )
    # x: array([[-3943658.715, 3363404.132, 3704651.343]])
    # y: array([[-3943658.709, 3363404.124, 3704651.307]])
    np.testing.assert_array_almost_equal(out_c, geocentric, decimal=1)

    out_d = idp.metashape.convert_proj3d(
        geocentric, geo_c, geo_d, is_geo=False
    )
    # x: array([[139.540336,  35.737563,  96.849   ]])
    # y: array([[139.540336,  35.737564,  96.878276]])
    np.testing.assert_array_almost_equal(out_d[0:2], geodetic[0:2], decimal=5)
    np.testing.assert_array_almost_equal(out_d[2], geodetic[2], decimal=1)


#########################
# test metashape object #
#########################
def test_class_init_metashape():
    m1 = idp.Metashape()
    assert m1.software == "metashape"

    # load with project_path
    m2 = idp.Metashape(project_path=os.path.join(ms_path, "goya_test.psx"))
    assert m2.project_name == "goya_test"
    assert m2.label == ''

    # load with project_path + chunk_id
    m2 = idp.Metashape(
        project_path=os.path.join(ms_path, "goya_test.psx"), chunk_id=0
    )
    assert m2.project_name == "goya_test"
    ## check values
    assert m2.label == 'Chunk 1'
    assert m2.meta == {}
    assert len(m2.photos) == 259
    assert m2.photos[0].label == "DJI_0284.JPG"
    assert m2.photos[0].enabled == False
    assert m2.photos[0].path == "//172.31.12.56/pgg2020a/drone/20201029/goya/DJI_0284.JPG"
    assert m2.photos[0].rotation.shape == (3, 3)
    assert m2.photos[0].sensor.width == m2.sensors[0].width

    # error init with chunk_id without project_path
    with pytest.raises(
        LookupError, 
        match=re.escape("Could not load chunk_id ")):
        m3 = idp.Metashape(chunk_id=0)


def test_local2world2local():
    attempt1 = idp.Metashape()
    attempt1.transform.matrix = np.asarray([
        [-0.86573098, -0.01489186,  0.08977677,  7.65034123],
        [ 0.06972335,  0.44334391,  0.74589315,  1.85910928],
        [-0.05848325,  0.74899678, -0.43972184, -0.1835615],
        [ 0.,          0.,          0.,           1.]], dtype=np.float)
    w_pos = np.array([0.5, 1, 1.5])
    l_pos = np.array(
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965]
    )
    w_pos_ans = np.array(
        [0.4999999999999978, 0.9999999999999993, 1.5]
    )

    world_pos = attempt1._local2world(l_pos)
    np.testing.assert_array_almost_equal(w_pos_ans, world_pos, decimal=6)

    local_pos = attempt1._world2local(w_pos)
    np.testing.assert_array_almost_equal(l_pos, local_pos, decimal=6)


def test_metashape_project_local_points_on_raw():
    test_project_folder = os.path.join(ms_path, "goya_test.psx")

    chunk = idp.Metashape(test_project_folder, chunk_id=0)

    # test for single point
    l_pos = np.array([7.960064093299587, 1.3019528769064523, -2.6697181763370965])

    p_dis_out = chunk.back2raw_single(
        l_pos, photo_id=0, distortion_correct=False)
    p_undis_out = chunk.back2raw_single(
        l_pos, photo_id=0, distortion_correct=True)

    # pro_api_out = np.asarray([2218.883386793118, 1991.4709388015149])
    my_undistort_out = np.array([2220.854889556147, 1992.6933680261686])
    my_distort_out = np.array([2218.47960556, 1992.46356322])

    np.testing.assert_array_almost_equal(p_dis_out, my_distort_out)
    np.testing.assert_array_almost_equal(p_undis_out, my_undistort_out)

    # test for multiple points
    l_pos_points = np.array([
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965],
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965]])

    p_dis_outs = chunk.back2raw_single(
        l_pos_points, photo_id=0, distortion_correct=False)
    p_undis_outs = chunk.back2raw_single(
        l_pos_points, photo_id=0, distortion_correct=True)

    my_undistort_outs = np.array([
        [2220.854889556147, 1992.6933680261686],
        [2220.854889556147, 1992.6933680261686]])
    my_distort_outs = np.array([
        [2218.47960556, 1992.46356322],
        [2218.47960556, 1992.46356322]])

    np.testing.assert_array_almost_equal(p_dis_outs, my_distort_outs)
    np.testing.assert_array_almost_equal(p_undis_outs, my_undistort_outs)


def test_world2crs_and_on_raw_images():
    test_project_folder = os.path.join(ms_path, "wheat_tanashi.psx")
    chunk = idp.Metashape(test_project_folder, chunk_id=0)

    local = np.array(
        [11.870130675203006, 0.858098777517136, -12.987136541275])
    geocentric = np.array(
        [-3943658.7087006606, 3363404.124223561, 3704651.3067566575])
    # columns=['lon', 'lat', 'alt']
    geodetic = np.array(
        [139.54033578028609, 35.73756358928734, 96.87827569602781])

    idp_world = chunk._local2world(local)
    np.testing.assert_array_almost_equal(idp_world, geocentric, decimal=1)

    idp_crs = chunk._world2crs(idp_world)
    np.testing.assert_array_almost_equal(idp_crs, geodetic)

    camera_id = 56
    camera_label = "DJI_0057"
    camera_pix_ans = np.array([2391.7104647010146, 1481.8987733175165])

    idp_cam_pix = chunk.back2raw_single(local, camera_id, distortion_correct=True)
    np.testing.assert_array_almost_equal(camera_pix_ans, idp_cam_pix)

    idp_cam_pix_l = chunk.back2raw_single(local, camera_label, distortion_correct=True)
    np.testing.assert_array_almost_equal(camera_pix_ans, idp_cam_pix_l)
