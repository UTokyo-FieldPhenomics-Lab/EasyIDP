import pyproj
import numpy as np
from easyric.io.geotiff import _prase_header_string, point_query, mean_values


def test_prase_header_string_width():
    out_dict = _prase_header_string("* 256 image_width (1H) 13503")
    assert out_dict['width'] == 13503


def test_prase_header_string_length():
    out_dict = _prase_header_string("* 257 image_length (1H) 19866")
    assert out_dict['length'] == 19866


def test_prase_header_string_scale():
    out_dict = _prase_header_string("* 33550 model_pixel_scale (3d) (0.0029700000000000004, 0.0029700000000000004, 0")
    assert out_dict['scale'] == (0.0029700000000000004, 0.0029700000000000004)


def test_prase_header_string_tie_point():
    out_dict = _prase_header_string("* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823,")
    assert out_dict['tie_point'] == (368090.77975000005, 3956071.13823)
    out_dict = _prase_header_string("* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823, 0")
    assert out_dict['tie_point'] == (368090.77975000005, 3956071.13823)


def test_prase_header_string_nodata():
    out_dict = _prase_header_string("* 42113 gdal_nodata (7s) b'-10000'")
    assert out_dict['nodata'] == -10000


def test_prase_header_string_proj_normal(capsys):
    out_dict = _prase_header_string("* 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'")
    captured = capsys.readouterr()
    assert captured.out == "[io][geotiff][GeoCorrd] Comprehense [* 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'] to geotiff coordinate tag [WGS 84 / UTM zone 54N]\n"
    assert out_dict['proj'] == pyproj.CRS.from_epsg(32654)


def test_prase_header_string_proj_error(capsys):
    # should raise error because WGS 84 / UTM ... should be full
    out_dict = _prase_header_string("* 34737 geo_ascii_params (30s) b'UTM zone 54N|WGS 84|'")
    captured = capsys.readouterr()
    assert '[io][geotiff][GeoCorrd] Generation failed, because [Input is not a CRS: UTM zone 54N]' in  captured.out
    assert out_dict['proj'] == None


def test_point_query_one_point():
    point = (368023.004, 3955500.669)
    out = point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', point)
    np.testing.assert_almost_equal(out, np.float32(97.45558), decimal=3)

def test_point_query_numpy_points():
    points = np.asarray([[368022.581, 3955501.054], [368024.032, 3955500.465]])
    out = point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', points)
    expected = np.asarray([97.624344, 97.59617])

    np.testing.assert_almost_equal(out, expected, decimal=3)


def test_point_query_list_numpy_points():
    points = np.asarray([[368022.581, 3955501.054], [368024.032, 3955500.465]])
    point = np.asarray([[368023.004, 3955500.669]])
    p_list = [point, points]

    expected = [np.asarray([97.45558]), np.asarray([97.624344, 97.59617])]
    out = point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', p_list)

    assert type(expected) == type(out)
    np.testing.assert_almost_equal(expected[0], out[0], decimal=3)
    np.testing.assert_almost_equal(expected[1], out[1], decimal=3)

def test_point_query_wrong_types():
    pass

def test_mean_values(capsys):
    mean_ht = mean_values(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif')
    captured = capsys.readouterr()
    assert mean_ht == np.float32(97.562584)