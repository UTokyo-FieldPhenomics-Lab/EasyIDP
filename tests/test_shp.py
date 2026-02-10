import pytest
import pyproj
import re
import numpy as np
import easyidp as idp

test_data = idp.data.TestData()

########################
# test show shp fields #
########################

complex_shp_review = (
    "  [-1] #            [0] ID                [1] MASSIFID       [2] CROPTYPE    [3] CROPDATE    [4] CROPAREA    [5] ATTID\n"
    "--------  ---------------------------  -------------------  --------------  --------------  --------------  -----------\n"
    "       0  230104112201809010000000000  2301041120000000000       小麦         2018-09-01     61525.26302\n"
    "       1  230104112201809010000000012  2301041120000000012       蔬菜         2018-09-01      2802.33512\n"
    "       2  230104112201809010000000014  2301041120000000014       玉米         2018-09-01      6960.7745\n"
    "     ...              ...                      ...               ...             ...             ...            ...\n"
    "     320  230104112201809010000000583  2301041120000000583       大豆         2018-09-01      380.41704\n"
    "     321  230104112201809010000000584  2301041120000000584       其它         2018-09-01      9133.25998\n"
    "     322  230104112201809010000000585  2301041120000000585       其它         2018-09-01      1704.27193\n"
)

roi_shp_preview = (
    "  [-1] #   [0] id\n"
    "--------  --------\n"
    "       0     0\n"
    "       1     1\n"
    "       2     2\n"
)

plots_shp_preview = (
    "  [-1] #   [0] plot_id\n"
    "--------  -------------\n"
    "       0      N1W1\n"
    "       1      N1W2\n"
    "       2      N1W3\n"
    "     ...       ...\n"
    "     109      S4E5\n"
    "     110      S4E6\n"
    "     111      S4E7\n"
)


def test_show_shp_fields_complex(capfd):
    idp.shp.show_shp_fields(test_data.shp.complex_shp, encoding="GBK")
    out, err = capfd.readouterr()

    # mac & windows give different space number
    assert out.replace(" ", "") == complex_shp_review.replace(" ", "")


def test_show_shp_fields_roi(capfd):
    idp.shp.show_shp_fields(test_data.shp.roi_shp)
    out, err = capfd.readouterr()
    assert out == roi_shp_preview


def test_show_shp_fields_plots(capfd):
    idp.shp.show_shp_fields(test_data.shp.lotus_shp)
    out, err = capfd.readouterr()
    assert out == plots_shp_preview


###########################
# test read_proj function #
###########################
def test_read_prj():
    prj_path = test_data.shp.roi_prj
    out_proj = idp.shp.read_proj(prj_path)

    check_proj = pyproj.CRS.from_string("WGS84 / UTM Zone 54N")
    assert out_proj.name == check_proj.name
    assert out_proj.coordinate_system == check_proj.coordinate_system


###########################
# test read_shp function #
###########################
def test_read_shp_without_target():
    # the degree unit (lat & lon) shp file using lon, lat order
    shp_path = test_data.shp.lonlat_shp

    # will raise error if shp_proj not given
    with pytest.raises(
        ValueError, match=re.escape("Unable to find the proj coordinate info [")
    ):
        lonlat_shp = idp.shp.read_shp(shp_path)

    # continue if shp_proj is given
    polygons, records, fields = idp.shp.read_shp(
        shp_path,
        shp_proj=pyproj.CRS.from_epsg(4326),
    )
    wanted_np = np.asarray(
        [
            [134.8312376, 34.90284972],
            [134.8312399, 34.90285097],
            [134.8312371, 34.90285516],
            [134.8312349, 34.90285426],
            [134.8312376, 34.90284972],
        ]
    )

    np.testing.assert_almost_equal(polygons[0], wanted_np)
    assert isinstance(records[0], dict)
    assert isinstance(fields, dict)


def test_read_shp_proj_success_print(capfd):
    shp_path = test_data.shp.testutm_shp
    polygons, records, fields = idp.shp.read_shp(shp_path)
    out, err = capfd.readouterr()
    assert (
        "[shp][proj] Use projection [WGS 84 / UTM zone 53N] for loaded shapefile [test_utm.shp]\n"
        in out
    )

    wanted_np = np.asarray(
        [
            [484593.62443893, 3862259.85857523],
            [484593.71826623, 3862259.85888695],
            [484593.71795452, 3862259.76537136],
            [484593.62319206, 3862259.76537136],
            [484593.62443893, 3862259.85857523],
        ]
    )

    np.testing.assert_almost_equal(polygons[0], wanted_np)
    assert isinstance(records[0], dict)
    assert isinstance(fields, dict)


def test_read_shp_show_duplicate():
    data = test_data.shp.rice_shp
    expected = [
        "NAR081_5",
        "NAR123_2",
        "NAR125_5",
        "SUP034_5",
        "Koshi3_1",
        "Koshi3_2",
        "Koshi3_3",
        "Koshi3_4",
        "Koshi3_5",
        "Koshi3_6",
        "3058#1_5",
        "Koshi_1",
        "15-135-2_1_6",
        "15-135-2_1_5",
        "15-135-2_1_4",
        "15-135-2_1_3",
        "15-135-2_1_2",
        "15-135-2_1_1",
        "15-135-2_2_1",
        "15-135-2_2_2",
        "15-135-2_2_3",
        "15-135-2_2_4",
        "15-135-2_2_5",
        "15-135-2_2_6",
    ]
    # missing = 3058#1_5 Koshi_1x6


def test_read_shp_non_exist_key_name_error():
    shp_path = test_data.shp.complex_shp

    polygons, records, fields = idp.shp.read_shp(shp_path, encoding="gbk")
    assert len(polygons) > 0
    assert len(records) > 0
    assert "CROPTYPE" in fields


def test_convert_shp():
    lonlat_path = test_data.shp.lonlat_shp
    utm_path = test_data.shp.utm53n_shp

    lonlat_polygons, _, _, lonlat_proj = idp.shp.read_shp(
        lonlat_path,
        shp_proj=pyproj.CRS.from_epsg(4326),
        return_proj=True,
    )

    utm_polygons, _, _, utm_proj = idp.shp.read_shp(utm_path, return_proj=True)

    lonlat_shp = {str(i): p for i, p in enumerate(lonlat_polygons)}
    utm_shp = {str(i): p for i, p in enumerate(utm_polygons)}

    utm_cvt_shp = idp.geotools.convert_proj(lonlat_shp, lonlat_proj, utm_proj)

    a = utm_cvt_shp["0"]
    b = utm_shp["0"]

    #  A         B
    #   o-------o
    #   |       |
    #   |       |
    #   |       |
    #   |       |
    #   o-------o
    #  C         D

    # lon_lat order: [A, B, C, D, A]
    # lonlat_utm order: [A, D. C. B, A]

    np.testing.assert_almost_equal(a, np.flip(b, axis=0), decimal=6)


def test_convert_xy_order_jp():
    # fix the https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/issues/104
    # testing by the Japanese Coordinate epsg 30166
    roi = idp.ROI(test_data.shp.jp_crs_shp, name_field="name")
    source_np_id_1 = np.array(
        [
            [7823.3030067, -112609.60232095],
            [7822.21310759, -112609.52507115],
            [7822.21128186, -112607.32159423],
            [7823.27253373, -112607.32505667],
            [7823.3030067, -112609.60232095],
        ]
    )

    np.testing.assert_almost_equal(roi["1"], source_np_id_1)

    convert_np_id_1 = np.array(
        [
            [136.08281125, 34.98802348],
            [136.08279931, 34.98802419],
            [136.08279931, 34.98804405],
            [136.08281094, 34.98804401],
            [136.08281125, 34.98802348],
        ]
    )

    roi.change_crs(pyproj.CRS.from_epsg(4326))
    np.testing.assert_almost_equal(roi["1"], convert_np_id_1)
