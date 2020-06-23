import easyric
import numpy as np
from easyric.objects.wrapper import EasyParams
from easyric.objects.software import Pix4dFiles
from easyric.calculation import external_internal_calc, get_img_name_and_coords

import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection

project_name  = r"wheat_hand_ukyoto_lowD1_20200427"
project_path = r"D:\OneDrive\Documents\4_PhD\14_[Paper]BroccoliPaper\Data\wheat_revert.demo"
params_path = f"{project_path}\params"
raw_img_folder = f"{project_path}\photos"
dom_path = f"{project_path}\\{project_name}_transparent_mosaic_group1.tif"
dsm_path = f"{project_path}\\{project_name}_dsm.tif"

points_txt = f"{project_path}/picking_list.txt"

center_points = np.loadtxt(points_txt, delimiter=',')
num = int(len(center_points) / 4)
center_point_list = []
for i in range(0, num):
    points = center_points[4*i:4*i+4, :]
    points = np.vstack([points, center_points[4*i, :][None,:]])
    center_point_list.append(points)

p4df = Pix4dFiles(project_path=project_path, raw_img_path=raw_img_folder, project_name=project_name)
p4df.manual_specify(param_folder=params_path, dom_path=dom_path, dsm_path=dsm_path)

proj = EasyParams()
proj.from_pix4d_files(p4df)

for i, center_point in enumerate(center_point_list):

    img_name, img_coords = easyric.calculation.get_img_name_and_coords(proj, center_point)

    for img_n, img_c in zip(img_name, img_coords):
        print(i, img_n)
        #'''
        fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=72)
        # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
        ax.imshow(imread(f"{raw_img_folder}/{img_n}")[::-1,:,:], origin="lower")

        polygon = pts.Polygon(img_c, True)
        p = PatchCollection([polygon], alpha=0.5, facecolors='red')

        ax.add_collection(p)
        ax.set_title(f"Reprojection on {img_n} via Ex/Internal files")
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.savefig(f"{project_path}/picking_list.result/id{i}-{img_n}.png")
        plt.close()
        #'''