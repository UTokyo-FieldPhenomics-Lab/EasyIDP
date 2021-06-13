import os

import numpy as np
import pytest
import easyidp
from easyidp.core.objects import ReconsProject, Points
from easyidp.io import metashape

module_path = os.path.join(easyidp.__path__[0], "io/tests")


def test_init_reconsproject():
    attempt1 = ReconsProject("agisoft")
    assert attempt1.software == "metashape"

    attempt2 = ReconsProject("Metashape")
    assert attempt2.software == "metashape"

    with pytest.raises(LookupError):
        attempt3 = ReconsProject("not_supported_sfm")


def test_local2world2local():
    attempt1 = ReconsProject("agisoft")
    attempt1.transform.matrix = np.asarray([[-0.86573098, -0.01489186, 0.08977677, 7.65034123],
                                            [0.06972335, 0.44334391, 0.74589315, 1.85910928],
                                            [-0.05848325, 0.74899678, -0.43972184, -0.1835615],
                                            [0., 0., 0., 1.]], dtype=np.float)
    w_pos = Points([0.5, 1, 1.5])
    l_pos = Points([7.960064093299587, 1.3019528769064523, -2.6697181763370965])
    w_pos_ans = Points([0.4999999999999978, 0.9999999999999993, 1.5])

    world_pos = attempt1.local2world(l_pos)
    np.testing.assert_array_almost_equal(w_pos_ans.values, world_pos.values, decimal=6)

    local_pos = attempt1.world2local(w_pos)
    np.testing.assert_array_almost_equal(l_pos.values, local_pos.values, decimal=6)


def test_metashape_project_local_points_on_raw():
    test_project_folder = easyidp.test_full_path("data/metashape/goya_test.psx")
    chunks = metashape.open_project(test_project_folder)

    chunk = chunks[0]

    # test for single point
    l_pos = Points([7.960064093299587, 1.3019528769064523, -2.6697181763370965])

    p_dis_out = chunk.project_local_points_on_raw(l_pos, 0, distortion_correct=False)
    p_undis_out = chunk.project_local_points_on_raw(l_pos, 0, distortion_correct=True)

    # pro_api_out = np.asarray([2218.883386793118, 1991.4709388015149])
    my_undistort_out = Points([2220.854889556147, 1992.6933680261686])
    my_distort_out = Points([2218.47960556, 1992.46356322])

    np.testing.assert_array_almost_equal(p_dis_out.values, my_distort_out.values)
    np.testing.assert_array_almost_equal(p_undis_out.values, my_undistort_out.values)

    # test for multiple points
    l_pos_points = Points([[7.960064093299587, 1.3019528769064523, -2.6697181763370965],
                           [7.960064093299587, 1.3019528769064523, -2.6697181763370965]])

    p_dis_outs = chunk.project_local_points_on_raw(l_pos_points, 0, distortion_correct=False)
    p_undis_outs = chunk.project_local_points_on_raw(l_pos_points, 0, distortion_correct=True)

    my_undistort_outs = Points([[2220.854889556147, 1992.6933680261686],
                                [2220.854889556147, 1992.6933680261686]])
    my_distort_outs = Points([[2218.47960556, 1992.46356322],
                              [2218.47960556, 1992.46356322]])

    np.testing.assert_array_almost_equal(p_dis_outs.values, my_distort_outs.values)
    np.testing.assert_array_almost_equal(p_undis_outs.values, my_undistort_outs.values)


def test_world2crs_and_on_raw_images():
    test_project_folder = easyidp.test_full_path("data/metashape/wheat_tanashi.psx")
    chunks = metashape.open_project(test_project_folder)

    chunk = chunks[0]

    local = Points([11.870130675203006, 0.858098777517136, -12.987136541275])
    geocentric = Points([-3943658.7087006606, 3363404.124223561, 3704651.3067566575])
    geodetic = Points([139.54033578028609, 35.73756358928734, 96.87827569602781], columns=['lon', 'lat', 'alt'])

    idp_world = chunk.local2world(local)
    np.testing.assert_array_almost_equal(idp_world.values, geocentric.values, decimal=1)

    idp_crs = chunk.world2crs(idp_world)
    np.testing.assert_array_almost_equal(idp_crs.values, geodetic.values)

    camera_id = 56   # camera_label = 'DJI_0057'
    camera_pix_ans = Points([2391.7104647010146, 1481.8987733175165])

    idp_cam_pix = chunk.project_local_points_on_raw(local, camera_id, distortion_correct=True)
    np.testing.assert_array_almost_equal(camera_pix_ans.values, idp_cam_pix.values)


