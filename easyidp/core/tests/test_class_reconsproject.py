import os

import numpy as np
import pytest
import easyidp
from easyidp.core.objects import ReconsProject
from easyidp.io import metashape

module_path = os.path.join(easyidp.__path__[0], "io/tests")


def test_init_reconsproject():
    attempt1 = ReconsProject("agisoft")
    assert attempt1.software == "metashape"

    attempt2 = ReconsProject("Metashape")
    assert attempt2.software == "metashape"

    with pytest.raises(LookupError):
        attempt3 = ReconsProject("not_supported_sfm")


def test_world2local2world():
    attempt1 = ReconsProject("agisoft")
    attempt1.transform.matrix = np.asarray([[-0.86573098, -0.01489186, 0.08977677, 7.65034123],
                                            [0.06972335, 0.44334391, 0.74589315, 1.85910928],
                                            [-0.05848325, 0.74899678, -0.43972184, -0.1835615],
                                            [0., 0., 0., 1.]], dtype=np.float)
    l_pos = np.asarray([0.5, 1, 1.5])
    w_pos = np.asarray([7.960064093299587, 1.3019528769064523, -2.6697181763370965])
    l_pos_ans = np.asarray([0.4999999999999978, 0.9999999999999993, 1.5])

    local_pos = attempt1.world2local(w_pos)
    np.testing.assert_array_almost_equal(l_pos_ans, local_pos, decimal=6)

    world_pos = attempt1.local2world(l_pos)
    np.testing.assert_array_almost_equal(w_pos, world_pos, decimal=6)


def test_metashape_project_world_points_on_raw():
    test_project_folder = easyidp.test_full_path("data/metashape/goya_test.psx")
    chunks = metashape.open_project(test_project_folder)

    chunk = chunks[0]

    # test for single point
    w_pos = np.asarray([7.960064093299587, 1.3019528769064523, -2.6697181763370965])

    p_dis_out = chunk.project_world_points_on_raw(w_pos, 0, distortion_correct=False)
    p_undis_out = chunk.project_world_points_on_raw(w_pos, 0, distortion_correct=True)

    pro_api_out = np.asarray([2218.883386793118, 1991.4709388015149])
    my_undistort_out = np.asarray([2220.854889556147, 1992.6933680261686])
    my_distort_out =np.asarray([2218.47960556, 1992.46356322])

    np.testing.assert_array_almost_equal(p_dis_out, my_distort_out)
    np.testing.assert_array_almost_equal(p_undis_out, my_undistort_out)

    # test for multiple points
    w_pos_points = np.asarray([[7.960064093299587, 1.3019528769064523, -2.6697181763370965],
                               [7.960064093299587, 1.3019528769064523, -2.6697181763370965]])

    p_dis_outs = chunk.project_world_points_on_raw(w_pos_points, 0, distortion_correct=False)
    p_undis_outs = chunk.project_world_points_on_raw(w_pos_points, 0, distortion_correct=True)

    my_undistort_outs = np.asarray([[2220.854889556147, 1992.6933680261686],
                                    [2220.854889556147, 1992.6933680261686]])
    my_distort_outs =np.asarray([[2218.47960556, 1992.46356322],
                                 [2218.47960556, 1992.46356322]])

    np.testing.assert_array_almost_equal(p_dis_outs, my_distort_outs)
    np.testing.assert_array_almost_equal(p_undis_outs, my_undistort_outs)