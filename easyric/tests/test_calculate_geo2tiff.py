from easyric.calculate import geo2tiff

def test_shp_clip_geotiff():
    out_dict = geo2tiff.shp_clip_geotiff(shp_path='file/shp_test/test.shp',
                                         geotiff_path='file/tiff_test/2_12.tif',
                                         out_folder='out/shp_clip')
    assert list(out_dict.keys()) == ['0.png']