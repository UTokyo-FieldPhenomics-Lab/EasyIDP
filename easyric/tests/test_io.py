import unittest
import numpy as np
import pyproj
from easyric.objects import Pix4D
from easyric.io import plot
from easyric.io.geotiff import _prase_header_string
from easyric.io.shp import read_proj
from easyric.calculation import external_internal_calc, get_img_name_and_coords


class TestGeotiff(unittest.TestCase):

    def test_prase_header_string_width(self):
        out_dict = _prase_header_string("* 256 image_width (1H) 13503")
        self.assertEqual(out_dict['width'], 13503)

    def test_prase_header_string_length(self):
        out_dict = _prase_header_string("* 257 image_length (1H) 19866")
        self.assertEqual(out_dict['length'], 19866)

    def test_prase_header_string_scale(self):
        out_dict = _prase_header_string("* 33550 model_pixel_scale (3d) (0.0029700000000000004, 0.0029700000000000004, 0")
        self.assertEqual(out_dict['scale'], (0.0029700000000000004, 0.0029700000000000004))

    def test_prase_header_string_tie_point(self):
        out_dict = _prase_header_string("* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823,")
        self.assertEqual(out_dict['tie_point'], (368090.77975000005, 3956071.13823))
        out_dict = _prase_header_string("* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823, 0")
        self.assertEqual(out_dict['tie_point'], (368090.77975000005, 3956071.13823))

    def test_prase_header_string_nodata(self):
        out_dict = _prase_header_string("* 42113 gdal_nodata (7s) b'-10000'")
        self.assertEqual(out_dict['nodata'], -10000)

    def test_prase_header_string_proj_normal(self):
        out_dict = _prase_header_string("* 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'")
        self.assertEqual(out_dict['proj'], pyproj.CRS.from_epsg(32654))

    def test_prase_header_string_proj_error(self):
        # should raise error because WGS 84 / UTM ... should be full
        out_dict = _prase_header_string("* 34737 geo_ascii_params (30s) b'UTM zone 54N|WGS 84|'")
        self.assertEqual(out_dict['proj'], None)

class TestShp(unittest.TestCase):

    def test_read_prj(self):
        prj_path = r'file/pix4d.diy/roi.prj'
        out_proj = read_proj(prj_path)

        check_proj = pyproj.CRS.from_string('WGS84 / UTM Zone 54N')
        self.assertEqual(out_proj.name, check_proj.name)
        self.assertEqual(out_proj.coordinate_system, check_proj.coordinate_system)

class TestPlot(unittest.TestCase):

    def test_plot_img(self):
        center_points = np.asarray([[0.91523622, -0.77530369, 0.47057343],
                                    [0.40791818, -1.28262173, 0.49362946],
                                    [0.21158767, -0.56954931, 0.49823761],
                                    [0.91523622, -0.77530369, 0.47057343]])

        project_name = f"broccoli_tanashi_5_20191008_mavicRGB_15m_M"
        project_path = f"D:/OneDrive/Documents/4_PhD/10_Data/02_UAV.Broccoli.Pix4D/2019tanashi_broccoli5_AP_new/20191008"
        params_path = f"{project_path}"
        raw_img_folder = f'{project_path}/photos'

        test_photo = "DJI_0209.JPG"

        p4d = Pix4D(project_path=project_path, raw_img_path=raw_img_folder, project_name=project_name, param_folder=params_path)

        coords_exin = external_internal_calc(p4d, center_points, test_photo)

        expected_out = [[1178.1021297262118, 3105.3147746182062],
                        [994.5716006627708, 3244.3760643458304],
                        [964.0093214752706, 3013.9080158878337],
                        [1178.1021297262118, 3105.3147746182062]]
        self.assertEqual(coords_exin.tolist(), expected_out)

        # test popup one img
        plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=True)

        # test save one img
        plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=False,
                                 file_name=f'out/test_1_{test_photo}.png')

        # test save imgs
        img_names, img_coords = get_img_name_and_coords(p4d, points=center_points, method='exin')
        plot.draw_polygon_on_imgs(p4d, img_names[0:2], img_coords[0:2], out_folder='out', coord_prefix='test_multi')

if __name__ == '__main__':
    unittest.main()
