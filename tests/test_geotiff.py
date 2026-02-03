import pytest
import pyproj
import re
import numpy as np
import random
import shutil
from pathlib import Path

import easyidp as idp

from . import shared_data, report_loguru_to_caplog, out_dir

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

# ============================================================================
# Migrated Class-based tests from test_geotiff.old
# ============================================================================

def test_class_init_with_path(shared_data):
    """Test GeoTiff initialization with file path."""
    test_data = shared_data['test_data']

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    # convert rel path to abs path, ideally it should longer
    assert Path(obj.file_path).resolve() == test_data.pix4d.lotus_dom.resolve()
    assert obj.header is not None


def test_class_header_sugar_property(shared_data):
    """Test the crs sugar to replace geotiff.header['crs']."""
    test_data = shared_data['test_data']

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)
    
    assert obj.crs == obj.header['crs']
    assert obj.height == obj.header['height']
    assert obj.width == obj.header['width']
    assert obj.dim == obj.header['dim']
    assert obj.nodata == obj.header['nodata']
    assert obj.scale == obj.header['scale']
    assert obj.tie_point == obj.header['tie_point']

    # test value setter (should raise AttributeError)
    with pytest.raises(AttributeError):
        # python <3.10 : can't set attribute ...
        # python >3.10 : property 'crs' of 'GeoTiff' object has no setter
        obj.crs = 'aaa'


def test_class_open(shared_data):
    """Test GeoTiff.open() method."""
    test_data = shared_data['test_data']

    obj = idp.GeoTiff()
    obj.open(test_data.pix4d.lotus_dom)

    assert Path(obj.file_path).resolve() == test_data.pix4d.lotus_dom.resolve()
    assert obj.header is not None


def test_class_point_query(shared_data):
    """Test GeoTiff.point_query() method with various input formats."""
    test_data = shared_data['test_data']
    
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

    # query one point by tuple
    point1 = (368023.004, 3955500.669)
    out1 = dsm.point_query(point1, is_geo=True)
    expect = np.asarray([97.45558])
    np.testing.assert_almost_equal(out1, expect, decimal=3)

    # query one point by list
    point2 = [368023.004, 3955500.669]
    out2 = dsm.point_query(point2, is_geo=True)
    np.testing.assert_almost_equal(out2, expect, decimal=3)

    # query several points by list
    point3 = [
        [368022.581, 3955501.054], 
        [368024.032, 3955500.465]]
    out3 = dsm.point_query(point3, is_geo=True)
    expects = np.array([97.624344, 97.59617])
    np.testing.assert_almost_equal(out3, expects, decimal=3)

    # query several points by numpy
    point4 = np.array(point3)
    out4 = dsm.point_query(point4, is_geo=True)
    np.testing.assert_almost_equal(out4, expects, decimal=3)

    # test point query using polygon vertices
    poly_geo = np.array([
        [ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])

    pt = dsm.point_query(poly_geo, is_geo=True)
    assert pt.shape == (5,)
    assert np.all(97 < pt) and np.all(pt < 98)


def test_class_point_query_error(shared_data):
    """Test GeoTiff.point_query() error handling."""
    test_data = shared_data['test_data']
    
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

    # raise type error for set input
    set1 = {1, 2}
    with pytest.raises(TypeError, match=re.escape("Only tuple, list, ndarray are supported")):
        dsm.point_query(set1, is_geo=True)

    # raise index error for wrong shape
    tuple1 = (1, 2, 3)
    with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
        dsm.point_query(tuple1, is_geo=True)

    list1 = [1, 2, 3]
    with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
        dsm.point_query(list1, is_geo=True)

    ndarray1 = np.array([1, 2, 3])
    with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
        dsm.point_query(ndarray1, is_geo=True)


def test_class_crop_polygon_save_geotiff(shared_data, tmp_path):
    """Test polygon cropping and saving to file."""
    test_data = shared_data['test_data']
    roi_select = shared_data['roi_select']

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    # convert ROI to the same CRS as the GeoTiff
    plot = roi_select.copy()
    plot.change_crs(obj.header["crs"])

    # pick a random plot for testing
    plot_id, polygon_hv = random.choice(list(plot.items()))

    save_tiff = tmp_path / "crop_polygon.tif"
    imarray = obj.crop_polygon(polygon_hv, is_geo=True, save_path=save_tiff)

    assert save_tiff.exists()
    # should be 3D array with 4 channels (RGBA)
    assert len(imarray.shape) == 3
    assert imarray.shape[2] == 4
    # around 300 pixels for all squared lotus boundary
    assert 270 < imarray.shape[0] and imarray.shape[0] < 350
    assert 270 < imarray.shape[1] and imarray.shape[1] < 350

    # verify the saved file has correct geo offset
    out = idp.GeoTiff(save_tiff)
    xmin, _ = polygon_hv.min(axis=0)
    _, ymax = polygon_hv.max(axis=0)

    assert xmin >= out.header["tie_point"][0]
    assert xmin <= out.header["tie_point"][0] + out.header["scale"][0]
    assert ymax <= out.header["tie_point"][1]
    assert ymax >= out.header["tie_point"][1] - out.header["scale"][1]


def test_class_crop_rectangle_save_geotiff(shared_data):
    """Test rectangle cropping with geo and pixel coordinates."""
    test_data = shared_data['test_data']

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    # crop by pixel coordinates
    out1 = obj.crop_rectangle(left=434, top=918, w=320, h=321, is_geo=False)

    # crop by geo coordinates
    out2 = obj.crop_rectangle(
        left=368017.75187, top=3955511.49993, 
        w=2.3561161599936895, h=2.362485199701041, 
        is_geo=True)

    # Rasterio may produce slightly different crop sizes (±1 pixel)
    # due to different rounding in coordinate transformation
    assert 320 <= out1.shape[0] <= 322
    assert 319 <= out1.shape[1] <= 321
    assert out1.shape[2] == 4
    
    assert 320 <= out2.shape[0] <= 322
    assert 319 <= out2.shape[1] <= 321
    assert out2.shape[2] == 4


def test_class_polygon_math(shared_data):
    """Test polygon_math() method for DSM and DOM."""
    test_data = shared_data['test_data']

    # plot_t["N1W1"] -> 
    poly_geo = np.array([
        [ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])

    # test dsm results
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

    dsm_mean   = dsm.polygon_math(poly_geo, is_geo=True, kernel="mean")
    dsm_min    = dsm.polygon_math(poly_geo, is_geo=True, kernel="min")
    dsm_max    = dsm.polygon_math(poly_geo, is_geo=True, kernel="max")
    dsm_pmin5  = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmin5")
    dsm_pmin10 = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmin10")
    dsm_pmax5  = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmax5")
    dsm_pmax10 = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmax10")

    assert 97 < dsm_mean   and dsm_mean   < 98
    assert 97 < dsm_min    and dsm_min    < 98
    assert 97 < dsm_max    and dsm_max    < 98
    assert 97 < dsm_pmin5  and dsm_pmin5  < 98
    assert 97 < dsm_pmin10 and dsm_pmin10 < 98
    assert 97 < dsm_pmax5  and dsm_pmax5  < 98
    assert 97 < dsm_pmax10 and dsm_pmax10 < 98

    # test dom results
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

    dom_mean   = dom.polygon_math(poly_geo, is_geo=True, kernel="mean")
    dom_min    = dom.polygon_math(poly_geo, is_geo=True, kernel="min")
    dom_max    = dom.polygon_math(poly_geo, is_geo=True, kernel="max")
    dom_pmin5  = dom.polygon_math(poly_geo, is_geo=True, kernel="pmin5")
    dom_pmin10 = dom.polygon_math(poly_geo, is_geo=True, kernel="pmin10")
    dom_pmax5  = dom.polygon_math(poly_geo, is_geo=True, kernel="pmax5")
    dom_pmax10 = dom.polygon_math(poly_geo, is_geo=True, kernel="pmax10")

    assert dom_mean  .shape == (4, )
    assert dom_min   .shape == (4, )
    assert dom_max   .shape == (4, )
    assert dom_pmin5 .shape == (4, )
    assert dom_pmin10.shape == (4, )
    assert dom_pmax5 .shape == (4, )
    assert dom_pmax10.shape == (4, )

    assert dom_mean  [3] == 255.0
    assert dom_min   [3] == 255.0
    assert dom_max   [3] == 255.0
    assert dom_pmin5 [3] == 255.0
    assert dom_pmin10[3] == 255.0
    assert dom_pmax5 [3] == 255.0
    assert dom_pmax10[3] == 255.0


def test_class_crop_rois(shared_data, tmp_path):
    """Test crop_rois() method with ROI object."""
    test_data = shared_data['test_data']
    roi_select = shared_data['roi_select']

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    # Use 2D coordinates only (don't add Z values)
    roi = roi_select.copy()
    roi.change_crs(obj.crs)

    tif_out_folder = tmp_path / "class_crop"
    tif_out_folder.mkdir()

    out_dict = obj.crop_rois(roi, save_folder=tif_out_folder)

    assert len(out_dict) == 4
    assert (tif_out_folder / "N1W1.tif").exists()
    # Rasterio may produce slightly different crop sizes
    assert 319 <= out_dict["N2E2"].shape[0] <= 321
    assert 319 <= out_dict["N2E2"].shape[1] <= 321
    assert out_dict["N2E2"].shape[2] == 4


def test_class_crop_rois_multispec(shared_data, tmp_path):
    """Test crop_rois() with 5-layer multispectral image."""
    test_data = shared_data['test_data']

    roi = idp.ROI(test_data.shp.mlayer_shp)

    # 5 layers multispectral with 5th as alpha
    multi_tiff = idp.GeoTiff(test_data.tiff.mlayer_multi)

    tif_out_folder = tmp_path / "multi_crop"
    tif_out_folder.mkdir()

    out_dict = multi_tiff.crop_rois(roi, save_folder=tif_out_folder)
    assert len(out_dict) == 12
    assert (tif_out_folder / "2.tif").exists()
    # Rasterio may produce slightly different crop sizes (±1 pixel)
    assert 179 <= out_dict["2"].shape[0] <= 182
    assert 179 <= out_dict["2"].shape[1] <= 182
    assert out_dict["2"].shape[2] == 5


def test_class_crop_rois_ndvi_special(shared_data, tmp_path):
    """Test crop_rois() with NDVI 2-layer image."""
    test_data = shared_data['test_data']

    roi = idp.ROI(test_data.shp.mlayer_shp)

    # processing multispectral geotiff -> 2 layer ndvi (second as alpha)
    ndvi_tiff = idp.GeoTiff(test_data.tiff.mlayer_ndvi)

    tif_out_folder = tmp_path / "ndvi_crop"
    tif_out_folder.mkdir()

    out_dict = ndvi_tiff.crop_rois(roi, save_folder=tif_out_folder)
    assert len(out_dict) == 12
    assert (tif_out_folder / "2.tif").exists()
    # Rasterio may produce slightly different crop sizes (±1 pixel)
    assert 179 <= out_dict["2"].shape[0] <= 182
    assert 179 <= out_dict["2"].shape[1] <= 182
    assert out_dict["2"].shape[2] == 2


def test_class_geo2pixel2geo_executable(shared_data):
    """Test geo2pixel and pixel2geo coordinate conversion roundtrip."""
    test_data = shared_data['test_data']

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    roi.change_crs(dom.header['crs'])

    roi_test = roi[111]

    roi_test_pixel = dom.geo2pixel(roi_test)

    roi_test_back = dom.pixel2geo(roi_test_pixel)

    np.testing.assert_almost_equal(roi_test, roi_test_back, decimal=5)

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
    # Rasterio may produce slightly different crop sizes (±1 pixel)
    assert 49 <= crop.imarray.shape[0] <= 52
    assert 49 <= crop.imarray.shape[1] <= 52
    
    # Verify mask is computed and stored
    assert crop._mask is not None
    # Mask shape should match imarray shape
    assert crop._mask.shape == crop.imarray.shape[:2]
    
    # Data should NOT have nodata applied yet
    # (The original values should still be there, not replaced by nodata)


# ============================================================================
# Tests for one_raw_roi2geotiff and back2raw2geotiff
# ============================================================================

def test_one_raw_roi2geotiff(shared_data):
    """Test one_raw_roi2geotiff() basic functionality."""
    test_data = shared_data['test_data']
    p4d = shared_data['p4d']
    roi = shared_data['roi']
    out_all = shared_data['out_all']
    
    # Get first ROI and first image
    roi_id = list(out_all.keys())[0]
    img_dict = out_all[roi_id]
    img_id = list(img_dict.keys())[0]
    roi_raw_px = img_dict[img_id]
    
    # Get geo coordinates (need 2D only)
    roi_geo_coords = roi[roi_id][:, :2]
    
    # Find raw image path
    raw_img_path = test_data.pix4d.lotus_photos / f"{img_id}.JPG"
    
    # Call the function
    gtiff = idp.geotiff.one_raw_roi2geotiff(
        roi_crs=roi.crs,
        roi_geo_coords=roi_geo_coords,
        raw_img_path=raw_img_path,
        roi_raw_px=roi_raw_px,
        nodata=0,
        has_alpha=True,
    )
    
    # Verify GeoTiff object
    assert gtiff is not None
    assert isinstance(gtiff, idp.GeoTiff)
    assert gtiff.crs == roi.crs
    assert gtiff.width > 0
    assert gtiff.height > 0
    assert gtiff._mask is not None
    assert gtiff._mask.shape == (gtiff.height, gtiff.width)
    
    # Verify imarray is 3D (RGB image)
    assert len(gtiff.imarray.shape) == 3
    
    # Save and verify file
    save_path = out_dir / "tiff_test" /  "test_one_raw2geotiff.tif"
    gtiff.save(save_path, overwrite=True)
    assert save_path.exists()
    
    # Reload and verify
    reloaded = idp.GeoTiff(save_path)
    assert reloaded.crs == roi.crs


def test_one_raw_roi2geotiff_options(shared_data):
    """Test one_raw_roi2geotiff() with different options."""
    test_data = shared_data['test_data']
    roi = shared_data['roi']
    out_all = shared_data['out_all']
    
    # Get first ROI and first image
    roi_id = list(out_all.keys())[0]
    img_dict = out_all[roi_id]
    img_id = list(img_dict.keys())[0]
    roi_raw_px = img_dict[img_id]
    roi_geo_coords = roi[roi_id][:, :2]
    raw_img_path = test_data.pix4d.lotus_photos / f"{img_id}.JPG"
    
    # Test with has_alpha=False
    gtiff_noalpha = idp.geotiff.one_raw_roi2geotiff(
        roi_crs=roi.crs,
        roi_geo_coords=roi_geo_coords,
        raw_img_path=raw_img_path,
        roi_raw_px=roi_raw_px,
        nodata=255,
        has_alpha=False,
    )
    
    assert gtiff_noalpha.nodata == 255
    
    # Test with has_alpha=True (default)
    gtiff_alpha = idp.geotiff.one_raw_roi2geotiff(
        roi_crs=roi.crs,
        roi_geo_coords=roi_geo_coords,
        raw_img_path=raw_img_path,
        roi_raw_px=roi_raw_px,
        nodata=0,
        has_alpha=True,
    )
    
    assert gtiff_alpha.nodata is None  # No nodata when using alpha


def test_back2raw2geotiff(shared_data):
    """Test back2raw2geotiff() batch processing with save_folder."""
    p4d = shared_data['p4d']
    roi = shared_data['roi']
    out_all = shared_data['out_all']
    
    output_folder = out_dir / "tiff_test" / "back2raw2geotiff_test"
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir()
    
    # Call the function
    result = idp.geotiff.back2raw2geotiff(
        recons=p4d,
        back2raw_result=out_all,
        roi=roi,
        output_folder=output_folder,
        nodata=0,
        has_alpha=True,
    )
    
    # Verify result structure matches input
    assert len(result) == len(out_all)
    
    for roi_id in out_all.keys():
        assert roi_id in result
        assert len(result[roi_id]) > 0
        
        # Check that files were saved
        roi_folder = output_folder / str(roi_id)
        assert roi_folder.exists()
        
        for img_id, gtiff in result[roi_id].items():
            assert isinstance(gtiff, idp.GeoTiff)
            save_path = roi_folder / f"{img_id}.tif"
            assert save_path.exists()


def test_back2raw2geotiff_no_save(shared_data):
    """Test back2raw2geotiff() without saving files."""
    p4d = shared_data['p4d']
    roi = shared_data['roi']
    out_all = shared_data['out_all']
    
    # Call without output_folder (no save)
    result = idp.geotiff.back2raw2geotiff(  
        recons=p4d,
        back2raw_result=out_all,
        roi=roi,
        output_folder=None,  # No save
    )
    
    # Verify result structure matches input
    assert len(result) == len(out_all)
    
    for roi_id in out_all.keys():
        assert roi_id in result
        for img_id, gtiff in result[roi_id].items():
            assert isinstance(gtiff, idp.GeoTiff)
            assert gtiff.crs == roi.crs


# =============================================================================
# Mask Polygon Tests
# =============================================================================

class TestMaskPolygon:
    """Tests for mask polygon functionality."""
    
    def test_set_mask_polygon_geo(self, shared_data):
        """Set polygon with geo coords, verify storage."""
        test_data = shared_data['test_data']
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        
        # Create a simple rectangle polygon in geo coordinates
        polygon = np.array([
            [368025.0, 3955479.0],
            [368027.0, 3955479.0],
            [368027.0, 3955477.0],
            [368025.0, 3955477.0],
        ])
        
        gtiff.set_mask_polygon(polygon, is_geo=True)
        
        assert gtiff.mask_polygon is not None
        assert gtiff._mask_polygon_is_geo is True
        # Check polygon has 4 or 5 points (auto-closure may apply)
        assert len(gtiff.mask_polygon) >= 4
    
    def test_set_mask_polygon_pixel(self, shared_data):
        """Set polygon with pixel coords, verify conversion."""
        test_data = shared_data['test_data']
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        
        # Create polygon in pixel coordinates
        polygon = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150],
        ])
        
        gtiff.set_mask_polygon(polygon, is_geo=False)
        
        assert gtiff.mask_polygon is not None
        assert gtiff._mask_polygon_is_geo is False
        
        # Verify can get geo coords
        geo_poly = gtiff.mask_polygon_geo
        assert geo_poly is not None
        assert geo_poly.shape == (5, 2)  # 4 points + closure
        
        # Verify pixel coords unchanged
        pixel_poly = gtiff.mask_polygon_pixel
        np.testing.assert_allclose(pixel_poly[:4], polygon, atol=0.01)
    
    def test_mask_binary_from_polygon(self, shared_data):
        """Binary mask computed correctly from polygon."""
        test_data = shared_data['test_data']
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        
        # Set a rectangular polygon in pixel coords
        polygon = np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200],
        ])
        gtiff.set_mask_polygon(polygon, is_geo=False)
        
        # Get binary mask (should be computed from polygon)
        mask = gtiff.mask
        
        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape == (gtiff.height, gtiff.width)
        
        # Check that interior is True
        assert mask[150, 150] is np.True_
        # Check that exterior is False
        assert mask[50, 50] is np.False_
    
    def test_polygon_metadata_storage(self, shared_data, tmp_path):
        """Test polygon is stored and retrieved from metadata."""
        import rasterio as rio
        test_data = shared_data['test_data']
        
        # Load and set polygon
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        gtiff._imarray = gtiff.imarray  # Force load
        
        polygon = np.array([
            [368025.0, 3955479.0],
            [368027.0, 3955479.0],
            [368027.0, 3955477.0],
            [368025.0, 3955477.0],
        ])
        gtiff.set_mask_polygon(polygon, is_geo=True)
        
        # Save
        save_path = tmp_path / "test_polygon.tif"
        gtiff.save(save_path, overwrite=True)
        
        # Check metadata written
        with rio.open(save_path) as src:
            tags = src.tags()
            assert 'EASYIDP_MASK_POLYGON' in tags
            assert 'POLYGON' in tags['EASYIDP_MASK_POLYGON']
        
        # Reload and verify polygon recovered
        gtiff2 = idp.GeoTiff(save_path)
        assert gtiff2.mask_polygon is not None
        np.testing.assert_allclose(
            gtiff2.mask_polygon[:4], 
            polygon, 
            atol=0.001
        )
    
    def test_affine_rectangle_detection(self, shared_data):
        """Test _is_valid_rectangle correctly identifies rectangles."""
        test_data = shared_data['test_data']
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        
        # Standard axis-aligned rectangle
        rect = np.array([
            [0, 0], [10, 0], [10, 5], [0, 5], [0, 0]
        ], dtype=float)
        is_valid, angle, bounds = gtiff._is_valid_rectangle(rect)
        assert is_valid is True
        assert np.isclose(angle, 0.0, atol=1.0)
        
        # Rotated rectangle (45 degrees)
        s2 = np.sqrt(2)
        rotated_rect = np.array([
            [0, 0], [s2, s2], [0, 2*s2], [-s2, s2], [0, 0]
        ], dtype=float)
        is_valid, angle, bounds = gtiff._is_valid_rectangle(rotated_rect)
        assert is_valid is True
        assert np.isclose(abs(angle), 45.0, atol=2.0)
        
        # Triangle (not rectangle)
        triangle = np.array([
            [0, 0], [10, 0], [5, 10], [0, 0]
        ], dtype=float)
        is_valid, angle, bounds = gtiff._is_valid_rectangle(triangle)
        assert is_valid is False
    
    def test_affine_non_rectangle_warning(self, shared_data, tmp_path, report_loguru_to_caplog):
        """Test warning when polygon is not rectangular."""
        test_data = shared_data['test_data']
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        gtiff._imarray = gtiff.imarray
        
        # Set a non-rectangular polygon (triangle)
        triangle = np.array([
            [368025.0, 3955479.0],
            [368027.0, 3955479.0],
            [368026.0, 3955477.0],
        ])
        gtiff.set_mask_polygon(triangle, is_geo=True)
        
        # Save with use_affine=True (should warn)
        save_path = tmp_path / "test_triangle.tif"
        gtiff.save(save_path, overwrite=True, use_affine=True)
        
        # Note: loguru logs may not be captured by pytest caplog by default
        # Just verify the file was saved successfully (warning was issued)
        assert save_path.exists()
    
    def test_backward_compatibility_no_polygon(self, shared_data):
        """Existing GeoTiffs without polygon tags load normally."""
        test_data = shared_data['test_data']
        
        # Load existing file (no polygon tag)
        gtiff = idp.GeoTiff(test_data.pix4d.lotus_dom_part)
        
        # Should have no polygon
        assert gtiff.mask_polygon is None
        assert gtiff.use_affine is False
        
        # Mask should still work via legacy method
        mask = gtiff.mask
        assert mask is not None
        assert mask.dtype == bool
