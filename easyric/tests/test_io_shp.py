import pyproj
import pytest
import numpy as np
from easyric.io.shp import read_proj, read_shp2d, read_shp3d


def test_read_prj():
    prj_path = r'./file/shp_test/roi.prj'
    out_proj = read_proj(prj_path)

    check_proj = pyproj.CRS.from_string('WGS84 / UTM Zone 54N')
    assert out_proj.name == check_proj.name
    assert out_proj.coordinate_system == check_proj.coordinate_system

def test_read_shp2d_without_target():
    # the degree unit (lat & lon) shp file using lon, lat order
    lonlat_shp = read_shp2d(f"file/shp_test/lon_lat.shp")
    # flipped for pyproj which input is (lat, lon)
    wanted_np = np.asarray([[34.90284972, 134.8312376],
                            [34.90285097, 134.8312399],
                            [34.90285516, 134.8312371],
                            [34.90285426, 134.8312349]])

    assert (lonlat_shp['1_02'] == wanted_np).all()

def test_read_shp2d_with_target_without_prj_file(capsys):
    # it will raise could not fild .prj file print and ignore the target_proj
    lonlat_shp = read_shp2d(f"file/shp_test/lon_lat.shp", geotiff_proj=pyproj.CRS.from_epsg(32654))
    captured = capsys.readouterr()
    assert captured.out == "[io][shp][proj] could not find ESRI projection file file/shp_test/lon_lat.prj, could not operate auto-convention, Please convert projection system manually.\n"

def test_read_shp2d_with_target_with_prj_file(capsys):
    lonlat_shp = read_shp2d(f"file/shp_test/roi.shp", geotiff_proj=pyproj.CRS.from_epsg(32654))
    captured = capsys.readouterr()
    assert captured.out == '[io][shp][proj] find ESRI projection file file/shp_test/roi.prj, and successfully obtain projection cartesian\n'

def test_read_shp3d_without_target_mean():
    lonlat_z = read_shp3d(r'file/pix4d.diy/plots.shp', r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif',
                          get_z_by='mean', shp_proj=pyproj.CRS.from_epsg(4326))

def test_read_shp3d_without_target_local():
    with pytest.raises(ValueError) as execinfo:
        # it will not work because prj file give a wrong CRS string
        lonlat_z = read_shp3d(r'file/pix4d.diy/plots.shp', r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif',
                              get_z_by='local')

        assert "ValueError: Fail to convert points from " in str(execinfo.value)

def test_read_shp3d_with_given_proj():
    # it should work because given a wrong CRS string
    # [Todo] cost too much time to run
    lonlat_z = read_shp3d(r'file/pix4d.diy/plots.shp', r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif',
                          get_z_by='local', shp_proj=pyproj.CRS.from_epsg(4326))