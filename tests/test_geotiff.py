import pytest
import pyproj
import re
import numpy as np
import random
import shutil
from pathlib import Path

import easyidp as idp

from . import shared_data, report_loguru_to_caplog

def test_def_get_header(shared_data):
    test_data = shared_data['test_data']
    
    lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dom)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 4
    assert lotus_full["nodata"] == None
    assert lotus_full["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dsm)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 1
    assert lotus_full["nodata"] == -10000.0
    assert lotus_full["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_part = idp.geotiff.get_header(test_data.pix4d.lotus_dom_part)
    assert lotus_part["width"] == 437
    assert lotus_part["height"] == 444
    assert lotus_part["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_part["tie_point"][0] == 368024.0839
    assert lotus_part["tie_point"][1] == 3955479.7512

def test_def_get_imarray(shared_data):
    test_data = shared_data['test_data']
    maize_part_np = idp.geotiff.get_imarray(test_data.pix4d.maize_dom)
    assert maize_part_np.shape == (722, 836, 4)

    lh = idp.geotiff.get_header(test_data.pix4d.lotus_dom_part)
    lotus_part_np = idp.geotiff.get_imarray(test_data.pix4d.lotus_dom_part)
    assert lotus_part_np.shape == (lh["height"], lh["width"], lh["dim"])

def test_def_geo2pixel2geo_UTM():
    gis_coord = np.asarray([
        [ 484593.67474654, 3862259.42413431],
        [ 484593.41064743, 3862259.92582402],
        [ 484593.64841806, 3862260.06515117],
        [ 484593.93077419, 3862259.55455913],
        [ 484593.67474654, 3862259.42413431]])

    # example file
    # TIFF file: 200423_G_M600pro_transparent_mosaic_group1.tif, 411 MiB, little endian, bigtiff
    # please check v1.0 easyric.tests.test_io_geotiff.py line 114
    # > https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/blob/a3420bc7b1e0f1013411565cf0e66dd2d2ba5371/easyric/tests/test_io_geotiff.py#L114
    # to get the full string of this header
    # here we just use the extracted results
    header = {'width': 19436, 'height': 31255, 'dim':4, 
              'scale': [0.001, 0.001], 'nodata': None,
              'tie_point': [484576.70205, 3862285.5109300003], 
              'crs': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}

    expected_pixel_idx = np.array([
        [16972, 26086],
        [16708, 25585],
        [16946, 25445],
        [17228, 25956],
        [16972, 26086]])

    expected_pixel_flt = np.array([
        [16972.69654   , 26086.79569047],
        [16708.59742997, 25585.10598028],
        [16946.36805996, 25445.77883044],
        [17228.72418998, 25956.37087012],
        [16972.69654   , 26086.79569047]])

    # ==========================
    # 1. return pixel int index
    # ==========================
    pixel_coord_idx = idp.geotiff.geo2pixel(gis_coord, header, return_index=True)
    np.testing.assert_almost_equal(pixel_coord_idx, expected_pixel_idx)

    # if return index, will cause precision loss
    gis_revert_idx = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_idx, gis_coord, decimal=3)

    # ===============================
    # 2. return pixel float position
    # ===============================
    pixel_coord_flt = idp.geotiff.geo2pixel(gis_coord, header)
    np.testing.assert_almost_equal(pixel_coord_flt, expected_pixel_flt)

    # then convert back should have fewer precision loss
    # but seems still have some preoblem
    gis_revert_flt = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_flt, gis_coord, decimal=3)

def test_def_geo2pixel2geo_lonlat():
    # using the source: https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions/44
    gis_latlon_coord = np.array([
        [-80.83957435, 25.78354364],
        [-80.83947435, 25.78354364],
        [-80.83947435, 25.78344364],
        [-80.83957435, 25.78344364],
        [-80.83957435, 25.78354364]])

    header = {
        'height': 8748, 'width': 7941, 'dim': 1, 'nodata': -32767.0, 
        'scale': [3.49222000000852e-07, 3.1617399999982425e-07], 
        'tie_point': [-80.84039234705898, 25.784493471936425], 
        'crs': pyproj.CRS.from_epsg(4326)
    }

    expected_pixel = np.array([
        [2342.34114395, 3004.14308711],
        [2628.6919466 , 3004.14308711],
        [2628.6919466 , 3320.42462829],
        [2342.34114395, 3320.42462829],
        [2342.34114395, 3004.14308711]])

    out = idp.geotiff.geo2pixel(gis_latlon_coord, header)

    np.testing.assert_almost_equal(out, expected_pixel)

    back = idp.geotiff.pixel2geo(out, header)

    np.testing.assert_almost_equal(back, gis_latlon_coord, decimal=3)

def test_def_point_query():
    # query one point
    point1 = (368023.004, 3955500.669)
    # query one point list
    point2 = [368023.004, 3955500.669]
    # query several points
    point3 = [
        [368022.581, 3955501.054], 
        [368024.032, 3955500.465]]
    # query several points by numpy
    point4 = np.array(point3)

    header = idp.geotiff.get_header(test_data.pix4d.lotus_dsm)
    with tf.TiffFile(test_data.pix4d.lotus_dsm) as tif:
        page = tif.pages[0]

        # point 1
        out1 = idp.geotiff.point_query(page, point1, header)
        expect = np.asarray([97.45558])
        np.testing.assert_almost_equal(out1, expect, decimal=3)

        # point 2
        out2 = idp.geotiff.point_query(page, point2, header)
        np.testing.assert_almost_equal(out2, expect, decimal=3)

        # point 3
        out3 = idp.geotiff.point_query(page, point3, header)
        expects = np.array([97.624344, 97.59617])
        np.testing.assert_almost_equal(out3, expects, decimal=3)

        # point 4
        out4 = idp.geotiff.point_query(page, point4, header)
        np.testing.assert_almost_equal(out4, expects, decimal=3)

# ============================================================================
# Tests for nodata/mask handling
# ============================================================================

def test_data_type_detection(shared_data):
    """Test _get_data_type() method for detecting dsm/rgb/rgba/ms/msa."""
    test_data = shared_data['test_data']
    
    # DSM should be detected as 'dsm'
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
    assert dsm._get_data_type() == 'dsm'
    
    # DOM with 4 bands uint8 should be 'rgba'
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    assert dom._get_data_type() == 'rgba'


def test_dsm_nodata_save(shared_data, tmp_path):
    """Test DSM saves with nodata value -32767.0."""
    test_data = shared_data['test_data']
    
    # Create a DSM with some masked regions
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
    
    # Get a small crop to work with
    crop = dsm.crop_rectangle(left=100, top=100, w=50, h=50, is_geo=False, 
                              return_geotiff=True)
    
    # The crop should have a mask computed
    assert crop._mask is not None
    assert crop._mask.shape == (crop.height, crop.width)
    
    # Save and verify nodata is set
    save_path = tmp_path / "test_dsm.tif"
    crop.save(save_path, overwrite=True)
    
    # Reload and check nodata value
    reloaded = idp.GeoTiff(save_path)
    assert reloaded.header['nodata'] == -32767.0


def test_rgb_alpha_save(shared_data, tmp_path):
    """Test RGB saves as RGBA with alpha channel."""
    test_data = shared_data['test_data']
    
    # Load DOM (RGBA)
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    
    # Crop a region (this should compute mask)
    crop = dom.crop_rectangle(left=100, top=100, w=50, h=50, is_geo=False,
                              return_geotiff=True)
    
    # Verify mask is computed
    assert crop._mask is not None
    
    # Save and verify no nodata (uses alpha instead)
    save_path = tmp_path / "test_rgba.tif"
    crop.save(save_path, overwrite=True)
    
    reloaded = idp.GeoTiff(save_path)
    # RGBA should have 4 bands and no nodata
    assert reloaded.header['dim'] == 4
    assert reloaded.header['nodata'] is None


def test_ms_alpha_save(shared_data, tmp_path):
    """Test multispectral saves with added alpha channel."""
    # Create a synthetic 5-band multispectral image
    # shape: (height, width, bands) = (100, 100, 5)
    ms_data = np.random.randint(0, 255, size=(100, 100, 5), dtype=np.uint16)
    
    # Create header
    header = {
        'height': 100,
        'width': 100,
        'dim': 5,
        'dtype': np.dtype('uint16'),
        'nodata': None,
        'scale': [1.0, 1.0],
        'tie_point': [0.0, 0.0],
        'crs': None,
        'profile': {
            'driver': 'GTiff',
            'height': 100,
            'width': 100,
            'count': 5,
            'dtype': 'uint16',
        }
    }
    
    ms_geotiff = idp.GeoTiff(imarray=ms_data, header=header)
    
    # Set a mask to simulate invalid regions
    mask = np.ones((100, 100), dtype=bool)
    mask[0:20, 0:20] = False  # Mark top-left as invalid
    ms_geotiff._mask = mask
    
    # Save
    save_path = tmp_path / "test_ms.tif"
    ms_geotiff.save(save_path, overwrite=True)
    
    # Reload and check - should have 6 bands (5 + alpha)
    reloaded = idp.GeoTiff(save_path)
    assert reloaded.header['dim'] == 6


def test_crop_preserves_full_data(shared_data):
    """Test that crop_* preserves full rectangular data, only storing mask."""
    test_data = shared_data['test_data']
    
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    
    # Crop a region
    crop = dom.crop_rectangle(left=100, top=100, w=50, h=50, is_geo=False,
                              return_geotiff=True)
    
    # Verify full rectangular data is preserved
    assert crop.imarray.shape[:2] == (50, 50)
    
    # Verify mask is computed and stored
    assert crop._mask is not None
    assert crop._mask.shape == (50, 50)
    
    # Data should NOT have nodata applied yet
    # (The original values should still be there, not replaced by nodata)
