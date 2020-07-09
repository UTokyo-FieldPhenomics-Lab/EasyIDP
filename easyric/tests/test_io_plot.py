import numpy as np
from easyric.objects import Pix4D
from easyric.io import plot
from easyric.calculate.geo2raw import external_internal_calc, get_img_name_and_coords

def test_plot_img(capsys):
    center_points = np.asarray([[0.91523622, -0.77530369, 0.47057343],
                                [0.40791818, -1.28262173, 0.49362946],
                                [0.21158767, -0.56954931, 0.49823761],
                                [0.91523622, -0.77530369, 0.47057343]])

    project_name = f"broccoli_tanashi_5_20191008_mavicRGB_15m_M"
    project_path = f"D:/OneDrive/Documents/4_PhD/10_Data/02_UAV.Broccoli.Pix4D/2019tanashi_broccoli5_AP_new/20191008"
    params_path = f"{project_path}"
    raw_img_folder = f'{project_path}/photos'

    test_photo = "DJI_0209.JPG"

    p4d = Pix4D(project_path=project_path, raw_img_path=raw_img_folder, project_name=project_name,
                param_folder=params_path)

    captured = capsys.readouterr()
    assert captured.out == "[Init][Pix4D] No ply given, however find 'broccoli_tanashi_5_20191008_mavicRGB_15m_M_group1_densified_point_cloud.ply' at current project folder\n" \
                           "[Init][Pix4D] No dom given, however find 'broccoli_tanashi_5_20191008_mavicRGB_15m_M_transparent_mosaic_group1.tif' at current project folder\n" \
                           "[Init][Pix4D] No dsm given, however find 'broccoli_tanashi_5_20191008_mavicRGB_15m_M_dsm.tif' at current project folder\n"

    coords_exin = external_internal_calc(p4d, center_points, test_photo)

    expected_out = [[1178.1021297262118, 3105.3147746182062],
                    [994.5716006627708, 3244.3760643458304],
                    [964.0093214752706, 3013.9080158878337],
                    [1178.1021297262118, 3105.3147746182062]]

    assert coords_exin.tolist() == expected_out

    # test popup one img
    plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=True)

    # test save one img
    plot.draw_polygon_on_img(p4d, img_name=test_photo, img_coord=coords_exin, show=False,
                             file_name=f'out/test_1_{test_photo}.png')

    # test save imgs
    img_names, img_coords = get_img_name_and_coords(p4d, points=center_points, method='exin')
    plot.draw_polygon_on_imgs(p4d, img_names[0:2], img_coords[0:2], out_folder='out', coord_prefix='test_multi')
