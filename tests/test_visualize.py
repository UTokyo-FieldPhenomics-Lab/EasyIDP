import pytest
import numpy as np
import pyproj
import re
from pathlib import Path
import easyidp as idp

test_data = idp.data.TestData()
lotus = idp.data.Lotus()

##############################
# Test back2raw_single based #
##############################

def test_class_back2raw_single():
    # lotus example
    p4d = idp.Pix4D()
    param_folder = test_data.pix4d.lotus_param
    image_folder = test_data.pix4d.lotus_photos
    p4d.open_project(test_data.pix4d.lotus_folder, raw_img_folder=image_folder, param_folder=param_folder)
    
    #plot, proj = idp.shp.read_shp(r"./tests/data/pix4d/lotus_tanashi_full/plots.shp", name_field=0, return_proj=True)
    #plot_t = idp.geotools.convert_proj(plot, proj, p4d.crs)
    plot =  np.array([   # N1E1
        [ 368020.2974959 , 3955511.61264302,      97.56272272],
        [ 368022.24288365, 3955512.02973983,      97.56272272],
        [ 368022.65361232, 3955510.07798313,      97.56272272],
        [ 368020.69867274, 3955509.66725421,      97.56272272],
        [ 368020.2974959 , 3955511.61264302,      97.56272272]
    ])

    out_dict = p4d.back2raw_crs(plot, distort_correct=True)

    # plot figures
    img_name = "DJI_0198"
    photo = p4d.photos[img_name]
    idp.visualize.draw_polygon_on_img(
        img_name, photo.path, out_dict[img_name], show=False, 
        save_as=test_data.vis.out / "p4d_back2raw_single_view.png")
    
#==============================
# advanced wrapper for classes
#==============================
    
roi = idp.ROI(lotus.shp, name_field='plot_id')
roi.get_z_from_dsm(lotus.metashape.dsm)

def test_visualize_one_roi_on_img_p4d():
    p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                    raw_img_folder=lotus.photo,
                    param_folder=lotus.pix4d.param)

    img_dict_p4d = roi.back2raw(p4d)

    with pytest.raises(IndexError, match=re.escape("Could not find backward results of plot [N1W2] on image [aaa]")):
        p4d.show_roi_on_img(img_dict_p4d, 'N1W2', 'aaa')

    with pytest.raises(FileNotFoundError, match=re.escape("Could not find the image file [DJI_2233] in the Pix4D project")):
        img_dict_p4d['N1W1']['DJI_2233'] = None
        p4d.show_roi_on_img(img_dict_p4d, 'N1W1', 'DJI_2233')

    out = p4d.show_roi_on_img(
            img_dict_p4d, 'N1W1', "DJI_0500", title="AAAA", color='green', alpha=0.5, show=False,
            save_as=test_data.vis.out / "p4d_show_roi_on_img_diy.png")
    
    out = p4d.show_roi_on_img(
            img_dict_p4d, 'N1W2', show=False, title=["AAAA", "BBBB"],  color='green', alpha=0.5,
            save_as=test_data.vis.out / "p4d_show_one_roi_all.png")


def test_visualize_one_roi_on_img_ms():
    ms = idp.Metashape(lotus.metashape.project, chunk_id=0)

    img_dict_ms = roi.back2raw(ms)

    with pytest.raises(IndexError, match=re.escape("Could not find backward results of plot [N1W2] on image [aaa]")):
        ms.show_roi_on_img(img_dict_ms, 'N1W2', 'aaa')

    with pytest.raises(FileNotFoundError, match=re.escape("Could not find the image file [DJI_2233] in the Metashape project")):
        img_dict_ms['N1W1']['DJI_2233'] = None
        ms.show_roi_on_img(img_dict_ms, 'N1W1', 'DJI_2233')

    out = ms.show_roi_on_img(
            img_dict_ms, 'N1W2', "DJI_0500", title="AAAA", color='green', alpha=0.5, show=False,
            save_as=test_data.vis.out / "ms_show_roi_on_img_diy.png")
    
    out = ms.show_roi_on_img(
            img_dict_ms, 'N1W2', color='green', alpha=0.5, 
            show=False, title=["AAAA", "BBBB"], 
            save_as=test_data.vis.out / "ms_show_one_roi_all.png")
    


####################################
# Test draw_backward_one_roi based #
####################################

def test_draw_backward_one_roi():
    ms = idp.Metashape(lotus.metashape.project, chunk_id=0)

    img_dict_ms = roi.back2raw(ms)

    with pytest.warns(UserWarning, match=re.escape(
            "Expected title like ['title1', 'title2'], not given 'sdedf', using default title instead")):
        idp.visualize.draw_backward_one_roi(
            ms, img_dict_ms['N1W1'], buffer=40, title='sdedf',
            save_as=test_data.vis.out / "draw_backward_one_roi.png",
            color='blue', show=False
        )
