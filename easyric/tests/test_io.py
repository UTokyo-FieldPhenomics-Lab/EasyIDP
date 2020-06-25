import unittest
import numpy as np
from easyric.objects.software import Pix4D
from easyric.calculation import external_internal_calc, get_img_name_and_coords
from easyric.io import plot

class TestGeotiff(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

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
        print(coords_exin)

        # test popup one img
        plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=True)

        # test save one img
        plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=False,
                                 file_name=f'out/test_1_{test_photo}.png')

        # test save imgs
        img_names, img_coords = get_img_name_and_coords(p4d, points=center_points, method='exin')
        plot.draw_polygon_on_imgs(p4d, img_names[0:5], img_coords[0:5], out_folder='out', coord_prefix='test_multi')

if __name__ == '__main__':
    unittest.main()
