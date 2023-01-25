import re
import os
import pytest
import numpy as np
import easyidp as idp
import shutil

test_data = idp.data.TestData()

def test_class_recons_output_io():
    # init return None
    recons = idp.reconstruct.Recons()

    assert recons.pcd is None
    assert recons.dom is None
    assert recons.dsm is None

    # raise error if given false type
    # dom test
    with pytest.raises(TypeError, match=re.escape("Please either specify DOM file ")):
        recons.dom = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given DOM file")):
        recons.dom = "tests/not_exist.tiff"

    # dsm test
    with pytest.raises(TypeError, match=re.escape("Please either specify DSM file ")):
        recons.dsm = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given DSM file")):
        recons.dsm = "tests/not_exist.tiff"

    # pcd test
    with pytest.raises(TypeError, match=re.escape("Please either specify pointcloud file ")):
        recons.pcd = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given pointcloud file")):
        recons.pcd = "tests/not_exist.ply"

    # pass if give correct file
    pcd = test_data.pix4d.lotus_pcd
    dsm = test_data.pix4d.lotus_dsm
    dom = test_data.pix4d.lotus_dom

    recons.dom = dom
    recons.dsm = dsm
    recons.pcd = pcd

    assert isinstance(recons.dom, idp.GeoTiff)
    assert isinstance(recons.dsm, idp.GeoTiff)
    assert isinstance(recons.pcd, idp.PointCloud)


def test_class_sensor_in_img_boundary():
    # ignore = None, x, y
    # polygon in bounds / outof bounds
    s = idp.reconstruct.Sensor()
    s.width = 4000
    s.height = 3000

    # x and y all out of boundary
    polygon = np.array([
        [-30, 400],
        [400, -30],
        [700, 800],
        [5000, 2500],
        [2700, 6000],
    ])

    assert s.in_img_boundary(polygon) is None
    assert s.in_img_boundary(polygon, ignore="x") is None
    assert s.in_img_boundary(polygon, ignore="y") is None

    # ====================
    polygon_x_in = np.array([
        [30, 400],
        [400, -30],
        [700, 800],
        [3000, 2500],
        [2700, 6000],
    ])

    polygon_y_in = np.array([
        [-30, 400],
        [400, 30],
        [700, 800],
        [5000, 2500],
        [2700, 2000],
    ])

    ix = s.in_img_boundary(polygon_y_in, ignore="x")
    iy = s.in_img_boundary(polygon_x_in, ignore="y")

    ix_expect = np.array([
        [0, 400],
        [400, 30],
        [700, 800],
        [4000, 2500],
        [2700, 2000],
    ])
    
    iy_expect = np.array([
        [30, 400],
        [400, 0],
        [700, 800],
        [3000, 2500],
        [2700, 3000],
    ])

    np.testing.assert_almost_equal(ix, ix_expect)
    np.testing.assert_almost_equal(iy, iy_expect)


def test_class_calibration_calibrate_error():
    # ============ pix4d =============
    c1 = idp.reconstruct.Calibration()
    c1.software = "pix4d_e"
    c1.type = "frame"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c1.calibrate(1,2)

    c2 = idp.reconstruct.Calibration()
    c2.software = "pix4d_e"
    c2.type = "fisheye"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c2.calibrate(1,2)

    c3 = idp.reconstruct.Calibration()
    c3.software = "pix4d"
    c3.type = "frame"

    c4 = idp.reconstruct.Calibration()
    c4.software = "pix4d"
    c4.type = "fisheye"
    with pytest.raises(
        NotImplementedError, 
        match=re.escape("Can not calibrate camera type [fisheye],")):
        c4.calibrate(1,2)

    # ========== Metashape ===========
    c5 = idp.reconstruct.Calibration()
    c5.software = "metashape_e"
    c5.type = "frame"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c5.calibrate(1,2)

    c6 = idp.reconstruct.Calibration()
    c6.software = "metashape_e"
    c6.type = "fisheye"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c6.calibrate(1,2)

    c7 = idp.reconstruct.Calibration()
    c7.software = "metashape"
    c7.type = "frame"

    c8 = idp.reconstruct.Calibration()
    c8.software = "metashape"
    c8.type = "fisheye"
    with pytest.raises(
        NotImplementedError, 
        match=re.escape("Can not calibrate camera type [fisheye],")):
        c4.calibrate(1,2)


def test_class_container_photo():
    p1 = idp.reconstruct.Photo()
    p2 = idp.reconstruct.Photo()
    p1.id = 0
    p1.label = "aaa.jpg"
    p2.id = 1
    p2.label = "bbb.jpg"

    a = idp.Container()
    a[p1.id] = p1
    a[p2.id] = p2

    assert len(a) == 2
    assert a[0] == p1
    assert a["bbb.jpg"] == p2

def test_func_sort_img_by_distance_ms():
    # =================
    # metashape outputs
    # =================
    ms = idp.Metashape(project_path=test_data.metashape.lotus_psx, chunk_id=0)

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    # only pick 2 plots as testing data
    key_list = list(roi.keys())
    for key in key_list:
        if key not in ["N1W1", "N1W2"]:
            del roi[key]
    roi.get_z_from_dsm(test_data.metashape.lotus_dsm)

    poly = roi["N1W2"]
    ms.crs = roi.crs

    out_all = ms.back2raw(roi)

    # test one roi
    cam_pos = ms.get_photo_position()
    filter_3 = idp.reconstruct._sort_img_by_distance_one_roi(ms, out_all["N1W2"], roi["N1W2"], cam_pos, num=3)
    assert len(filter_3) == 3

    filter_3_dis_01 = idp.reconstruct._sort_img_by_distance_one_roi(ms, out_all["N1W2"], roi["N1W2"], cam_pos, distance_thresh=0.1, num=3)
    assert len(filter_3_dis_01) == 0

    # test all roi
    filter_3_all = idp.reconstruct.sort_img_by_distance(ms, out_all, roi, num=3)
    assert len(filter_3_all) == 2
    for v in filter_3_all.values():
        assert len(v) == 3

    # on self
    filter_3_all_self = ms.sort_img_by_distance(out_all, roi, num=3)
    assert len(filter_3_all_self) == 2

# =============
# pix4d outputs
# =============
lotus = idp.data.Lotus()

p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                raw_img_folder=lotus.photo,
                param_folder=lotus.pix4d.param)
ms = idp.Metashape(project_path=lotus.metashape.project, chunk_id=0)

roi = idp.ROI(lotus.shp, name_field=0)
# only pick 2 plots as testing data
roi = roi[0:3]
roi.get_z_from_dsm(lotus.pix4d.dsm)

out_all = p4d.back2raw(roi)

def test_func_sort_img_by_distance_p4d():
    cam_pos = p4d.get_photo_position()
    filter_3 = idp.reconstruct._sort_img_by_distance_one_roi(p4d, out_all["N1W2"], roi["N1W2"], cam_pos, num=3)
    assert len(filter_3) == 3

    filter_3_dis_01 = idp.reconstruct._sort_img_by_distance_one_roi(p4d, out_all["N1W2"], roi["N1W2"], cam_pos, distance_thresh=0.1, num=3)
    assert len(filter_3_dis_01) == 0

    # test all roi
    filter_3_all = idp.reconstruct.sort_img_by_distance(p4d, out_all, roi, num=3)
    assert len(filter_3_all) == 3
    for v in filter_3_all.values():
        assert len(v) == 3

    # on self
    filter_3_all_self = p4d.sort_img_by_distance(out_all, roi, num=3)
    assert len(filter_3_all_self) == 3


def test_func_save_back2raw_json_and_png():

    with pytest.raises(TypeError, match=re.escape("Only the string path is acceptable, not [12345 <class 'int'>]")):
        idp.reconstruct.save_back2raw_json_and_png(p4d, out_all, 12345)

    out_path = test_data.b2r.out / "def_path"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    idp.reconstruct.save_back2raw_json_and_png(p4d, out_all, out_path)

    file_list = os.listdir(str(out_path))
    assert len(file_list) == len(out_all) + 2
    assert "roi_image_order.json" in file_list
    assert "image_roi_order.json" in file_list

    roi_folder_list = os.listdir(str(out_path / "N1W1"))
    assert len(roi_folder_list) == len(out_all['N1W1'])
    assert "N1W1_DJI_0479.png" in roi_folder_list

# test on the other easy-to-use functions
def test_func_save_back2raw_json_and_png_other_func():

    # this is very time costy, often no need to run...

    # ms_out_path = test_data.b2r.out / "ms_back2raw"
    # if os.path.exists(ms_out_path):
    #     shutil.rmtree(ms_out_path)
    # ms_out = ms.back2raw(roi, save_folder=ms_out_path)

    # roi_folder_list1 = os.listdir(str(ms_out_path / "N1W1"))
    # assert len(roi_folder_list1) == len(ms_out['N1W1'])


    # p4d_out_path = test_data.b2r.out / "p4d_back2raw"
    # if os.path.exists(p4d_out_path):
    #     shutil.rmtree(p4d_out_path)
    # p4d_out = p4d.back2raw(roi, save_folder=p4d_out_path)

    # roi_folder_list2 = os.listdir(str(p4d_out_path / "N1W1"))
    # assert len(roi_folder_list2) == len(p4d_out['N1W1'])


    ms_sort_path = test_data.b2r.out / "ms_back2raw_sort"
    if os.path.exists(ms_sort_path):
        shutil.rmtree(ms_sort_path)
    ms.sort_img_by_distance(ms_out, roi, num=3, save_folder=ms_sort_path)

    roi_folder_list3 = os.listdir(str(ms_sort_path / "N1W1"))
    assert len(roi_folder_list3) == 3


    p4d_sort_path = test_data.b2r.out / "p4d_back2raw_sort"
    if os.path.exists(p4d_sort_path):
        shutil.rmtree(p4d_sort_path)
    p4d.sort_img_by_distance(p4d_out, roi, num=3, save_folder=p4d_sort_path)
    
    roi_folder_list4 = os.listdir(str(p4d_sort_path / "N1W1"))
    assert len(roi_folder_list4) == 3
