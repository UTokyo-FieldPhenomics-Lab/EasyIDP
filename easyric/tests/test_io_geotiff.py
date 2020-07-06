import pyproj
from easyric.io.geotiff import _prase_header_string


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