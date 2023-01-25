import pytest
import numpy as np
import pyproj
import re
from pathlib import Path
import easyidp as idp

test_data = idp.data.TestData()

def test_hidden_match_suffix():
    test_folder = test_data.pix4d.maize_empty / "2_densification" / "point_cloud"

    out1 = idp.pix4d._match_suffix(test_folder, "xyz")
    assert out1 is None

    out2 = idp.pix4d._match_suffix(test_folder, "ply")
    assert out2 == f"{test_folder}/aaa_empty.ply"

    out3 = idp.pix4d._match_suffix(test_folder, ["laz", "ply"])
    assert out3 == f"{test_folder}/aaa_empty.laz"

    out4 = idp.pix4d._match_suffix(test_folder, ["xyz", "laz", "ply"])
    assert out4 == f"{test_folder}/aaa_empty.laz"

    out4 = idp.pix4d._match_suffix(test_folder, ["xyz", "ply", "234"])
    assert out4 == f"{test_folder}/aaa_empty.ply"


def test_parse_p4d_project_structure():
    # normal cases
    test_folder1 = test_data.pix4d.maize_folder
    proj_name1 = "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"

    p4d1 = idp.pix4d.parse_p4d_project(test_folder1)
    assert p4d1["project_name"] == proj_name1
    assert p4d1["param"]["project_name"] == proj_name1
    assert Path(p4d1["pcd"]).resolve() == \
        (test_folder1 / "2_densification" / "point_cloud" / f"{proj_name1}_group1_densified_point_cloud.ply").resolve()
    assert Path(p4d1["dsm"]).resolve() == \
        (test_folder1 / "3_dsm_ortho" / "1_dsm" / f"{proj_name1}_dsm.tif").resolve()
    assert Path(p4d1["dom"]).resolve() == \
        (test_folder1 / "3_dsm_ortho" / "2_mosaic" / f"{proj_name1}_transparent_mosaic_group1.tif").resolve()

    # test pix4d folder with wrong output name
    test_folder2 = test_data.pix4d.maize_empty

    # wrong project_name force to find outputs by file format
    p4d4 = idp.pix4d.parse_p4d_project(str(test_folder2))   # test string as input
    assert Path(p4d4["pcd"]).resolve() == \
        (test_folder2 / "2_densification" / "point_cloud" / "aaa_empty.ply").resolve()
    assert Path(p4d4["dsm"]).resolve() == \
        (test_folder2 / "3_dsm_ortho" / "1_dsm" / "bbb_dsm.tif").resolve()
    assert Path(p4d4["dom"]).resolve() == \
        (test_folder2 / "3_dsm_ortho" / "2_mosaic" / "ccc_dom.tif").resolve()


def test_parse_p4d_project_warning():
    # can not find output file
    test_folder = str(test_data.pix4d.maize_noout)
    
    with pytest.warns(UserWarning, 
        match=re.escape("Unable to find any")
    ):
        p4d2 = idp.pix4d.parse_p4d_project(test_folder)


def test_parse_p4d_project_structure_error():
    # folder without 1_init
    proj_name1 = "hasu_tanashi_20170525_Ins1RGB_30m"

    with pytest.raises(FileNotFoundError, 
        match=re.escape(
            f"Can not find pix4d parameter in given project folder"
        )
    ):
        p4d = idp.pix4d.parse_p4d_project(
            test_data.pix4d.lotus_folder, proj_name1
        )

    # no paramter folder
    with pytest.raises(FileNotFoundError, 
        match=re.escape(f"Can not find pix4d parameter in given project folder")
    ):
        p4d = idp.pix4d.parse_p4d_project(test_data.pix4d.maize_noparam)

################
# parse params #
################
param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")

def test_parse_p4d_param_folder():
    param = idp.pix4d.parse_p4d_param_folder(param_folder)

    assert param["project_name"] == "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
    assert Path(param["xyz"]).name == "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_offset.xyz"


def test_read_params():
    param = idp.pix4d.parse_p4d_param_folder(param_folder)
    xyz = idp.pix4d.read_xyz(param["xyz"])

    np.testing.assert_almost_equal(xyz, np.array([368009.0, 3955854.0, 97.0]))

    pmat = idp.pix4d.read_pmat(param["pmat"])

    assert "DJI_0954.JPG" in pmat.keys()
    assert pmat["DJI_0954.JPG"].shape == (3,4)

    cicp = idp.pix4d.read_cicp(param["cicp"])
    assert cicp["w_mm"] == 17.49998592
    assert cicp["h_mm"] == 13.124989440000002
    assert cicp["F"] == 15.011754049345175
    assert len(cicp) == 10

    ccp = idp.pix4d.read_ccp(param["ccp"])
    assert "DJI_0954.JPG" in ccp.keys()
    assert "w" in ccp.keys()
    assert "h" in ccp.keys()
    assert "cam_matrix" in ccp["DJI_0954.JPG"].keys()
    assert "rad_distort" in ccp["DJI_0954.JPG"].keys()

    cam_local_pos = np.array(
        [21.54872206687879199194, 
        -29.58734160676452162875,
        30.85702810138878149360])   # == geo_pos - offset
    np.testing.assert_almost_equal(ccp["DJI_0954.JPG"]["cam_pos"], cam_local_pos)

    cam_geo_pos = idp.pix4d.read_campos_geo(param["campos"])
    assert "DJI_0954.JPG" in cam_geo_pos.keys()
    assert cam_geo_pos["DJI_0954.JPG"].shape == (3,)
    np.testing.assert_almost_equal(
        cam_geo_pos["DJI_0954.JPG"],
        np.array([368030.548722,3955824.412658,127.857028])
    )

    ssk = idp.pix4d.read_cam_ssk(param["ssk"])
    assert ssk["label"] == "FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456"
    assert ssk["type"] == "frame"
    assert ssk["pixel_size"] == [3.79774000000000011568, 3.79774000000000011568]
    assert ssk["image_size_in_pixels"] == [ccp["h"], ccp["w"]]
    assert ssk["orientation"] == 1
    assert ssk["photo_center_in_pixels"] == [1727.5, 2303.5]

    crs = idp.shp.read_proj(param["crs"])
    assert crs == pyproj.CRS.from_epsg(32654)


def test_class_read_renamed_project():
    p4d = idp.Pix4D()

    param_folder = test_data.pix4d.lotus_param
    image_folder = test_data.pix4d.lotus_photos
    p4d.open_project(test_data.pix4d.lotus_folder, raw_img_folder=image_folder, param_folder=param_folder)

    assert p4d.software == "pix4d"
    assert p4d.label == "hasu_tanashi_20170525_Ins1RGB_30m"
    assert p4d.crs == pyproj.CRS.from_epsg(32654)

    np.testing.assert_almost_equal(
        p4d.meta["p4d_offset"], 
        np.array([368043.0, 3955495.0, 98.0]))

    assert isinstance(p4d.sensors[0], idp.reconstruct.Sensor)
    assert p4d.sensors[0].label == "FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456"

    assert p4d.sensors[0].calibration.f == 3951.0935994102065
    assert p4d.sensors[0].focal_length == 15.005226206224117

    # assert p4d.photos[0].label == "DJI_0151.JPG"
    assert p4d.photos[0].label == "DJI_0151"
    assert p4d.photos[0].path == ''
    assert "photos" in str(p4d.photos["DJI_0174.JPG"].path)
    assert p4d.photos["DJI_0174.JPG"].cam_matrix.shape == (3,3)
    assert p4d.photos["DJI_0174.JPG"].rotation.shape == (3,3)
    assert p4d.photos["DJI_0174.JPG"].location.shape == (3,)
    assert p4d.photos["DJI_0174.JPG"].transform.shape == (3,4)


def test_class_read_default_project():
    p4d = idp.Pix4D()
    p4d.open_project(test_data.pix4d.maize_folder)

    np.testing.assert_almost_equal(p4d.pcd.offset, p4d.meta["p4d_offset"])

    np.testing.assert_almost_equal(
        p4d.pcd.points[0, :],  
        np.array([368028.738, 3955822.024, 97.449]),
        decimal=3)

def test_class_back2raw_single():
    # lotus example
    p4d = idp.Pix4D()
    param_folder = test_data.pix4d.lotus_param
    image_folder = test_data.pix4d.lotus_photos
    p4d.open_project(test_data.pix4d.lotus_folder, raw_img_folder=image_folder, param_folder=param_folder)
    
    #plot, proj = idp.shp.read_shp(r"./tests/data/pix4d/lotus_tanashi_full/plots.shp", name_field=0, return_proj=True)
    #plot_t = idp.geotools.convert_proj(plot, proj, p4d.crs)
    plot =  np.array([   # N1E1
        [ 368020.2974959 , 3955511.61264302,      97.56272272],
        [ 368022.24288365, 3955512.02973983,      97.56272272],
        [ 368022.65361232, 3955510.07798313,      97.56272272],
        [ 368020.69867274, 3955509.66725421,      97.56272272],
        [ 368020.2974959 , 3955511.61264302,      97.56272272]
    ])

    out_dict = p4d.back2raw_crs(plot, distort_correct=True)
    assert len(out_dict) == 39

    px_0177 = np.array([
        [ 137.10982937, 2359.55887614],
        [ 133.56116243, 2107.13954299],
        [ 384.767746  , 2097.05639105],
        [ 388.10993307, 2350.41225998],
        [ 137.10982937, 2359.55887614]])

    np.testing.assert_almost_equal(out_dict["DJI_0177"], px_0177)

def test_class_back2raw():
    lotus = idp.data.Lotus()

    p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                    raw_img_folder=lotus.photo,
                    param_folder=lotus.pix4d.param)

    roi = idp.ROI(lotus.shp, name_field=0)
    # only pick 2 plots as testing data
    key_list = list(roi.keys())
    for key in key_list:
        if key not in ["N1W1", "N1W2"]:
            del roi[key]
    roi.get_z_from_dsm(lotus.pix4d.dsm)

    out_all = p4d.back2raw(roi)

    assert len(out_all) == 2

def test_class_back2raw_error():

    p4d = idp.Pix4D(test_data.pix4d.lotus_folder, 
                    test_data.pix4d.lotus_photos,
                    test_data.pix4d.lotus_param)

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)

    with pytest.raises(
        ValueError, 
        match=re.escape(
            "The back2raw function requires 3D roi with shape=(n, 3), but [N1W1] is (5, 2)")):
            p4d.back2raw(roi)


def test_class_get_photo_position():
    lotus = idp.data.Lotus()

    p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                    raw_img_folder=lotus.photo,
                    param_folder=lotus.pix4d.param)

    out = p4d.get_photo_position()

    assert len(out) == 151
    assert "DJI_0430" in out.keys()
    np.testing.assert_almost_equal(out['DJI_0430'], np.array([ 368020.07181613, 3955477.75605109,     136.75217778]))

    # convert to another proj?
    out_lonlat = p4d.get_photo_position(to_crs=pyproj.CRS.from_epsg(4326), refresh=True)
    assert len(out_lonlat) == 151
    assert "DJI_0430" in out_lonlat.keys()
    np.testing.assert_almost_equal(out_lonlat['DJI_0430'], np.array([139.5405607 ,  35.73445188, 136.75217778]))

    # change crs and refresh
    p4d.crs = pyproj.CRS.from_epsg(4326)
    assert p4d._photo_position_cache is None
    out_utm = p4d.get_photo_position()
    np.testing.assert_almost_equal(out_utm['DJI_0430'], np.array([139.5405607 ,  35.73445188, 136.75217778]))


def test_class_init():

    # test read folder & p4d file extention

    # easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d
    p4d = idp.Pix4D(test_data.pix4d.maize_folder)
    assert p4d.label == 'maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d'

    # easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d.p4d
    p4d = idp.Pix4D(str(test_data.pix4d.maize_folder) + '.p4d' )
    assert p4d.label == 'maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d'

def test_class_photos_get_by_short_name():
    p4d = idp.Pix4D()

    param_folder = test_data.pix4d.lotus_param
    image_folder = test_data.pix4d.lotus_photos
    p4d.open_project(test_data.pix4d.lotus_folder, raw_img_folder=image_folder, param_folder=param_folder)

    assert p4d.photos["DJI_0174"] == p4d.photos["DJI_0174.JPG"]