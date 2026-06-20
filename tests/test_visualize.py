import pytest
import numpy as np
import pyproj
import re
from pathlib import Path
import easyidp as idp

from . import shared_data

##############################
# Test back2raw_single based #
##############################

def test_class_back2raw_single(shared_data):
    test_data = shared_data["test_data"]

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

def test_visualize_one_roi_on_img_p4d(shared_data):
    test_data = shared_data["test_data"]
    roi = shared_data["roi_vis"]

    # p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
    #                 raw_img_folder=lotus.photo,
    #                 param_folder=lotus.pix4d.param)
    p4d = idp.Pix4D(project_path=test_data.pix4d.lotus_folder, 
                raw_img_folder=test_data.pix4d.lotus_photos,
                param_folder=test_data.pix4d.lotus_param)

    img_dict_p4d = roi.back2raw(p4d)

    with pytest.raises(IndexError, match=re.escape("Could not find backward results of plot [N1W2] on image [aaa]")):
        p4d.show_roi_on_img(img_dict_p4d, 'N1W2', 'aaa')

    # with pytest.raises(FileNotFoundError, match=re.escape("Could not find the image file [DJI_2233] in the Pix4D project")):
    #     # img_dict_p4d['N1W1']['DJI_2233'] = None
    #     p4d.show_roi_on_img(img_dict_p4d, 'N1W2', 'DJI_2233')

    out = p4d.show_roi_on_img(
            img_dict_p4d, 'N1W1', "DJI_0198", title="AAAA", color='green', alpha=0.5, show=False,
            save_as=test_data.vis.out / "p4d_show_roi_on_img_diy.png")
    
    out = p4d.show_roi_on_img(
            img_dict_p4d, 'N1W1', show=False, title=["AAAA", "BBBB"],  color='green', alpha=0.5,
            save_as=test_data.vis.out / "p4d_show_one_roi_all.png")


def test_visualize_one_roi_on_img_ms(shared_data):
    test_data = shared_data["test_data"]
    roi = shared_data["roi_vis"]

    ms = idp.Metashape(
        test_data.metashape.lotus_psx, 
        chunk_id=0, 
    )

    img_dict_ms = roi.back2raw(ms)

    with pytest.raises(IndexError, match=re.escape("Could not find backward results of plot [N1W2] on image [aaa]")):
        ms.show_roi_on_img(img_dict_ms, 'N1W2', 'aaa')

    # with pytest.raises(FileNotFoundError, match=re.escape("Could not find the image file [DJI_2233] in the Metashape project")):
    #     img_dict_ms['N1W1']['DJI_2233'] = None
    #     ms.show_roi_on_img(img_dict_ms, 'N1W1', 'DJI_2233')

    out = ms.show_roi_on_img(
            img_dict_ms, 'N1W1', "DJI_0500", title="AAAA", color='green', alpha=0.5, show=False,
            save_as=test_data.vis.out / "ms_show_roi_on_img_diy.png")
    
    out = ms.show_roi_on_img(
            img_dict_ms, 'N1W1', color='green', alpha=0.5, 
            show=False, title=["AAAA", "BBBB"], 
            save_as=test_data.vis.out / "ms_show_one_roi_all.png")
    


####################################
# Test draw_backward_one_roi based #
####################################

def test_draw_backward_one_roi(shared_data, report_logging_to_caplog):
    test_data = shared_data["test_data"]
    roi = shared_data["roi_vis"]
    
    # ms = idp.Metashape(lotus.metashape.project, chunk_id=0)
    ms = idp.Metashape(test_data.metashape.lotus_psx, chunk_id=0)

    img_dict_ms = roi.back2raw(ms)

    idp.visualize.draw_backward_one_roi(
        ms, img_dict_ms['N1W1'], buffer=40, title='sdedf',
        save_as=test_data.vis.out / "draw_backward_one_roi.png",
        color='blue', show=False
    )
    
    # Check that warning was logged via logging
    assert "Expected title like ['title1', 'title2']" in report_logging_to_caplog.text


######################
# Test show_subplots #
######################

OUTPUT_DIR = Path("tests/out/visual_test")

class TestShowSubplots:
    """Tests for subplot visualization output."""

    @pytest.fixture
    def setup_out_dir(self):
        """Ensure output directory exists."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    @pytest.fixture
    def rectangular_boundary(self):
        """Create a simple rectangular boundary ROI for testing."""
        roi = idp.ROI()
        # 10x20 meter rectangle
        roi["test_boundary"] = np.array([
            [0, 0],
            [20, 0],
            [20, 10],
            [0, 10],
            [0, 0],
        ], dtype=float)
        roi.crs = pyproj.CRS.from_epsg(32654)  # UTM 54N
        return roi

    @pytest.fixture
    def l_shaped_boundary(self):
        """Create an L-shaped boundary for testing non-rectangular cases."""
        roi = idp.ROI()
        # L-shape: 20x10 with 10x5 cut out from top-right
        roi["test_l_boundary"] = np.array([
            [0, 0],
            [20, 0],
            [20, 5],
            [10, 5],
            [10, 10],
            [0, 10],
            [0, 0],
        ], dtype=float)
        roi.crs = pyproj.CRS.from_epsg(32654)
        return roi

    def test_visualize_rect_grid(self, rectangular_boundary, setup_out_dir):
        """Test visualization of rectangular boundary grid."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=4,
            col_num=6,
            x_interval=0.5,
            y_interval=0.5
        )
        
        save_path = setup_out_dir / "rect_grid.png"
        idp.visualize.show_subplots(
            rectangular_boundary, 
            subplots, 
            title="Rectangular Boundary (4x6 grid)",
            save_as=str(save_path),
            show=False
        )
        assert save_path.exists()

    def test_visualize_l_shape_keep_all(self, l_shaped_boundary, setup_out_dir):
        """Test visualization of L-shaped boundary with keep='all'."""
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=4,
            col_num=6,
            keep="all"
        )
        
        save_path = setup_out_dir / "l_shape_keep_all.png"
        idp.visualize.show_subplots(
            l_shaped_boundary, 
            subplots, 
            title="L-Shape Boundary (keep='all')",
            save_as=str(save_path),
            show=False
        )
        assert save_path.exists()

    def test_visualize_l_shape_keep_touch(self, l_shaped_boundary, setup_out_dir):
        """Test visualization of L-shaped boundary with keep='touch'."""
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=4,
            col_num=6,
            keep="touch"
        )
        
        save_path = setup_out_dir / "l_shape_keep_touch.png"
        idp.visualize.show_subplots(
            l_shaped_boundary, 
            subplots, 
            title="L-Shape Boundary (keep='touch')",
            save_as=str(save_path),
            show=False
        )
        assert save_path.exists()

    def test_visualize_l_shape_keep_inside(self, l_shaped_boundary, setup_out_dir):
        """Test visualization of L-shaped boundary with keep='inside'."""
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=4,
            col_num=6,
            keep="inside"
        )
        
        save_path = setup_out_dir / "l_shape_keep_inside.png"
        idp.visualize.show_subplots(
            l_shaped_boundary, 
            subplots, 
            title="L-Shape Boundary (keep='inside')",
            save_as=str(save_path),
            show=False
        )
        assert save_path.exists()
