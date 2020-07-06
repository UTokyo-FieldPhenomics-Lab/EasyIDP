import pyproj
from easyric.io.shp import read_proj, read_shp2d, read_shp3d


def test_read_prj():
    prj_path = r'./file/pix4d.diy/roi.prj'
    out_proj = read_proj(prj_path)

    check_proj = pyproj.CRS.from_string('WGS84 / UTM Zone 54N')
    assert out_proj.name == check_proj.name
    assert out_proj.coordinate_system == check_proj.coordinate_system

def test_read_shp2d_without_target():
    pass
    #lonlat_shp = read_shp2d(f"file/shp_test/lon_lat.shp")
    #assert lonlat_shp['']