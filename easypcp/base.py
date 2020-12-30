import os
import imageio
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.cluster import KMeans
from skimage.measure import regionprops

from scipy.stats import gaussian_kde

from easypcp.pcd_tools import (pcd2binary,
                               pcd2voxel,
                               calculate_xyz_volume,
                               get_convex_hull,
                               build_cut_boundary)
from easypcp.geometry.min_bounding_rect import min_bounding_rect
from easypcp.io.cprint import printYellow
from easypcp.io.folder import make_dir
from easypcp.io.pcd import read_ply, read_plys
from easypcp.io.shp import read_shp, read_shps
from easypcp.plotting.figure import draw_3d_results, draw_plot_seg_results


class Classifier(object):
    """
    Variable:
        list path_list
        list kind_list
        set  kind_set
        skln clf
    """

    def __init__(self, path_list, kind_list, core='dtc', unit='m'):
        """
        :param path_list: the list training png path
            e.g. path_list = ['fore.png', 'back.png']
            or 'xxxx.ply'
            or 'o3d.geometry.PointCloud object'

        :param kind_list: list for related kind for path_list
            -1 is background
             0 is foreground (class 0)
            + 1 is class 1, etc. +

            Example 1: one class with 2 training data
                path_list = ['fore1.png', 'fore2.png']
                kind_list = [0, 0]
            Example 2: two class with 1 training data respectively
                path_list = ['back.png', 'fore.png']
                kind_list = [-1, 0]
            Example 2: multi class with multi training data
                path_list = ['back1.png', 'back2.png', 'leaf1.png', 'leaf2.png', 'flower1.png']
                kind_list = [-1, -1, 0, 0, 1]

        :param core:
            svm: Support Vector Machine Classifier
            dtc: Decision Tree Classifier
        """
        # Check whether correct input
        print('[Pnt][Classifier] Start building classifier')
        path_n = len(path_list)
        kind_n = len(kind_list)

        if path_n != kind_n:
            print('[Pnt][Classifier][Warning] the image number and kind number not matching!')

        self.path_list = path_list[0:min(path_n, kind_n)]
        self.kind_list = kind_list[0:min(path_n, kind_n)]

        # Build Training Array
        self.train_data = np.empty((0, 5))
        self.train_kind = np.empty(0)
        self.unit = unit
        self.build_training_array()
        print('[Pnt][Classifier] Training data prepared')

        self.kind_set = set(kind_list)

        if len(self.kind_set) == 1:   # only one class
            self.clf = OneClassSVM()
            # todo: build_svm1class()
        else:  # multi-classes
            if core == 'dtc':
                self.clf = DecisionTreeClassifier(max_depth=20)
                self.clf = self.clf.fit(self.train_data, self.train_kind)
            elif core == 'svm':
                self.clf = SVC()
                # todo: build SVC() classifier
        print('[Pnt][Classifier] Classifying model built')

    @staticmethod
    def read_png(file_path):
        img_ndarray = imageio.imread(file_path)
        h, w, d = img_ndarray.shape
        img_2d = img_ndarray.reshape(h * w, d)
        img_np = img_2d[img_2d[:, 3] == 255, 0:3] / 255

        return img_np

    @staticmethod
    def get_tgi(rgb_np):
        # -0.5 * [0.19(R-G) - 0.12(R-B)]
        tgi_np = -0.5 * (0.19 * (rgb_np[:,0] - rgb_np[:, 1]) - 0.12 * (rgb_np[:,0] - rgb_np[:, 2]))
        return tgi_np.reshape(tgi_np.shape[0], 1)

    def build_training_array(self):
        for img_path, kind in zip(self.path_list, self.kind_list):
            img_np = None
            if isinstance(img_path, o3d.geometry.PointCloud):
                img_np_rgb = np.asarray(img_path.colors)
                img_np_z = np.asarray(img_path.points)[:, 2].reshape(img_np_rgb.shape[0], 1)
                img_np_tgi = self.get_tgi(img_np_rgb)
                img_np = np.hstack([img_np_rgb, img_np_z, img_np_tgi])
            elif isinstance(img_path, str):
                if '.png' in img_path:
                    img_np_rgb = self.read_png(img_path)
                    img_np_z = np.ones((img_np_rgb.shape[0],1)) * -10000
                    img_np_tgi = self.get_tgi(img_np_rgb)
                    img_np = np.hstack([img_np_rgb, img_np_z, img_np_tgi])
                elif '.ply' in img_path:
                    pcd = read_ply(img_path, unit=self.unit)
                    img_np_rgb = np.asarray(pcd.colors)
                    img_np_z = np.asarray(pcd.points)[:, 2].reshape(img_np_rgb.shape[0], 1)
                    img_np_tgi = self.get_tgi(img_np_rgb)
                    img_np = np.hstack([img_np_rgb, img_np_z, img_np_tgi])
            else:
                raise TypeError(f"{img_path} is not supported, please only using png and ply files")

            kind_np = np.array([kind] * img_np.shape[0])
            self.train_data = np.vstack([self.train_data, img_np])
            self.train_kind = np.hstack([self.train_kind, kind_np])

    def predict(self, data):
        return self.clf.predict(data)


class Plot(object):
    """
    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    | The vector information are not loaded currently |
    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    Variables:
        pcd -> open3d.geometry.pointclouds object

        pcd_xyz = np.asarray(pcd.points) -> numpy.array nx3 object

        pcd_rgb = np.asarray(pcd.colors) -> numpy.array nx3 object

        x_max, y_min, z_len: # coordinate information for point clouds

        pcd_classified -> dict
            {'-1': o3d.geometry.pointclouds, # background
              '0': o3d.geometry.pointclouds, # foreground
              '1': o3d.geometry.pointclouds, # (optional) foreground 2
              ...etc.}

        pcd_denoised -> same as pcd_dict
            {'-1': o3d.geometry.pointclouds, # background denoised
              '0': o3d.geometry.pointclouds, # foreground denoised
              '1': o3d.geometry.pointclouds, # (optional) foreground 2 denoised
              ...etc.}

        pcd_segmented -> similar with pcd_dict
            {'0': [o3d.geometry.pointclouds, o3d.geometry.pointclouds, ...],  # background -1 not included
             '1': [o3d.geometry.pointclouds, o3d.geometry.pointclouds, ...].  # (optional)
             ... etc. }

        pcd_segmented_name
            {'0': ['class[0]-plant0', 'class[0]-plant1', ...],
             '1': ['class[1]-plant0', 'class[1]-plant1', ...]]}

    Functions:
        classifier_apply: apply the specified classifier.
            [params]
                clf: the Classifier class
            [return]
                self.pcd_classified

        remove_noise: apply the noise filtering algorithms to both background and foregrounds
            [param]

            [return]
                self.pcd_denoised

        auto_segmentation: segment the foreground by automatic point clouds algorithm (DBSCAN)
            please note, this may include some noises depends on how the classifier and auto-denoise parameters
            [param]
                denoise: whether remove some outlier points which has high possibility to be noises not removed
            [return]
                self.pcd_segmented

        shp_segmentation: segment the foreground by given shp file instead of using auto segmentation.
            [param]
                shp_dir: the path of shp file
            [return]
                self.pcd_segmented
    """

    def __init__(self, ply_path, clf, unit='m', output_path='.', write_ply=False, down_sample=True):
        self.ply_path = ply_path
        self.write_ply = write_ply

        # file I/O
        if os.path.isfile(ply_path):
            self.pcd = read_ply(ply_path, unit=unit)
            self.folder, tail = os.path.split(os.path.abspath(self.ply_path))
            self.ply_name = tail[:-4]
        elif os.path.isdir(ply_path):
            list_dir = os.listdir(ply_path)
            ply_list = []
            for item in list_dir:
                item_full_path = os.path.join(ply_path, item)
                if os.path.isfile(item_full_path) and '.ply' in item:
                    ply_list.append(item_full_path)
            if len(ply_list) == 0:
                raise EOFError(f'[{ply_path}] has no ply file')
            self.pcd = read_plys(ply_list, unit=unit)
            self.folder = os.path.abspath(ply_path)
            if ply_path[-1] in ['/', '\\']:
                self.ply_name = os.path.basename(ply_path[:-1])
            else:
                self.ply_name = os.path.basename(ply_path)
        else:
            raise TypeError(f'[{ply_path}] is neither a ply file or folder')

        print(f'[Pnt][Plot][__init__] Ply file "{self.ply_path}" loaded')

        # down sample check
        if down_sample:
            self.pcd = self.down_sample(self.pcd, part=100)

        if self.write_ply:
            if self.ply_name == '':
                raise IOError('Empty ply_name variable')
            self.out_folder = os.path.join(output_path, self.ply_name)
            print(f'[Pnt][Plot][__init__] Setting output folder "{os.path.abspath(self.out_folder)}"')
            make_dir(self.out_folder, clean=True)
        else:
            self.out_folder = output_path
            print(f'[Pnt][Plot][__init__] Mode "write_ply" == False, output folder creating ignored')

        self.pcd_xyz = np.asarray(self.pcd.points)
        self.pcd_rgb = np.asarray(self.pcd.colors)

        self.pcd_classified = self.classifier_apply(clf)

        self.segmented = False
        self.pcd_segmented = {}
        self.pcd_segmented_name = {}
        self.cov_warning = {}

    def classifier_apply(self, clf):
        print('[Pnt][Plot][Classifier_apply] Start Classifying')
        pcd_z = self.pcd_xyz[:, 2].reshape(self.pcd_xyz.shape[0],1)
        pcd_tgi = clf.get_tgi(self.pcd_rgb)
        input_np = np.hstack([self.pcd_rgb, pcd_z, pcd_tgi])
        pred_result = clf.predict(input_np)

        pcd_classified = {}

        for k in clf.kind_set:
            print(f'[Pnt][Plot][Classifier_apply] |-- classify class {k}')
            indices = np.where(pred_result == k)[0].tolist()
            pcd_classified[k] = self.pcd.select_by_index(indices=indices)
            # save ply
            if self.write_ply:
                o3d.io.write_point_cloud(os.path.join(self.out_folder, f'class[{k}].ply'),
                                         pcd_classified[k])
                print(f'[Pnt][Plot][Classifier_apply] |   |-- save to {self.out_folder}/class[{k}].ply')
            else:
                print(f'[Pnt][Plot][Classifier_apply] |   |-- mode "write_ply" == False, ply file not saved.')

        return pcd_classified

    def remove_noise(self, divide=100):
        # currently not recommend to use for sparse plant pcd, has removed from default __init__ steps.
        # # suitable for sfm -> single plants, which has large point numbers, delete some of them doesn't
        #            effect too much;
        # not suitable for plot level, each plant only have few points, may loss too much information
        pcd_voxel, voxel_params = pcd2voxel(self.pcd, part=divide)
        voxel_size, voxel_density = voxel_params['voxel_size'], voxel_params['voxel_density']

        pcd_cleaned = {}
        pcd_cleaned_id = {}
        print('[Pnt][Plot][remove_noise] Remove noises')
        for k in self.pcd_classified.keys():
            if k == -1:   # for background, need to apply statistical outlier removal
                cleaned, indices = self.pcd_classified[-1].remove_statistical_outlier(
                    nb_neighbors=round(voxel_density),
                    std_ratio=0.01)
                pcd_cleaned[-1], pcd_cleaned_id[-1] = cleaned.remove_radius_outlier(
                    nb_points=round(voxel_density*2),
                    radius=voxel_size)
            else:
                pcd_cleaned[k], pcd_cleaned_id[k] = self.pcd_classified[k].remove_radius_outlier(
                    nb_points=round(voxel_density),
                    radius=voxel_size)
            # save ply
            if self.write_ply:
                o3d.io.write_point_cloud(os.path.join(self.out_folder, f'class[{k}]-rm_noise.ply'),
                                         pcd_cleaned[k])
                print(f'[Pnt][Plot][remove_noise] ply file class[{k}]-rm_noise.ply saved to {self.out_folder}')
            else:
                print(f'[Pnt][Plot][remove_noise] Mode "write_ply" == False, ply file not saved.')
            # todo: add kde of ground points, and remove noises very close to ground points
            print(f'[Pnt][Plot][remove_noise] Kind {k} noise removed')

        self.pcd_classified = pcd_cleaned
        return pcd_cleaned   # , pcd_cleaned_id

    def auto_dbscan_args(self, eps_grids=10, divide=100):
        # split the shortest axis into 100 parts
        # the dbscan eps is the length of 10 grids
        # the min_points is the mean points of each grids (voxels)
        pcd_voxel, voxel_params = pcd2voxel(self.pcd, part=divide)
        voxel_size, voxel_density = voxel_params['voxel_size'], voxel_params['voxel_density']
        eps = voxel_size * eps_grids
        min_points = round(voxel_density)
        print(f'[Pnt][Plot][DBSCAN_Args] Recommend use eps={eps}, min_points={min_points} based on point density.')
        return eps, min_points

    def down_sample(self, pcd, part):
        # check whether need down-sampling
        _, voxel_params = pcd2voxel(pcd, part=part)
        voxel_size, voxel_density = voxel_params['voxel_size'], voxel_params['voxel_density']
        min_points = round(voxel_density)
        if min_points > 20:
            print(f'[Pnt][Plot][Down_Sample] Point cloud {self.ply_name} has average point counts [{min_points}] '
                  f'in the cube whose size={round(voxel_size*1000, 2)}mm.')
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size/5)   # 2^3=8, 3^3=27, 2.7^3=19.68
            _, voxel_params_down = pcd2voxel(pcd_down, voxel_size=voxel_size)
            voxel_size_down, voxel_density_down = voxel_params_down['voxel_size'], voxel_params_down['voxel_density']
            print(f'                        |--- Down sample to average counts [{round(voxel_density_down)}] ')
            return pcd_down
        else:
            return pcd

    def dbscan_segment(self, eps, min_points, pcd_dict=None):
        if pcd_dict is None:
            seg_in = self.pcd_classified
        else:
            seg_in = pcd_dict

        seg_out = {}
        for k in seg_in.keys():
            if k == -1:
                continue   # skip the background

            print(f'[Pnt][Plot][DBSCAN_Segment] Start segmenting class {k} Please wait...')
            vect = seg_in[k].cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
            vect_np = np.asarray(vect)
            seg_id = np.unique(vect_np)

            print(f'\n[Pnt][Plot][DBSCAN_Segment] Class {k} Segmented to {len(seg_id)} parts')
            pcd_seg_list = []
            pcd_seg_num = []
            for i, seg in enumerate(seg_id):
                indices = np.where(vect_np == seg)[0].tolist()
                pcd_seg = seg_in[k].select_by_index(indices)
                pcd_seg_list.append(pcd_seg)
                pcd_seg_num.append(len(indices))

            seg_out[k] = pcd_seg_list

            # coefficient of variance check to judge if need KMeans remove noise
            # [10000,13000,15000] -> 0.16222142113076257
            # [10000,13000,15000,12, 100]) -> 0.8369722427881723
            x = np.asarray(pcd_seg_num)
            cov = x.std() / x.mean()
            if cov >= 0.3:
                printYellow(f'[Warning] The coefficient of variance of '
                            f'point numbers too large ({round(cov,3)}), may contain noises!')
                if len(pcd_seg_num) < 20:
                    printYellow(f'{pcd_seg_num}')
                else:
                    printYellow(f"[{str(pcd_seg_num[:10])[1:-1]}, ..., {str(pcd_seg_num[-10:])[1:-1]}]")
                printYellow(f'Please consider use kmeans_split() to remove outlier noises.')
                self.cov_warning = True

        self.segmented = True
        self.pcd_segmented = seg_out
        return seg_out

    def kmeans_split(self, pcd_dict=None):
        if pcd_dict is None:
            split_in = self.pcd_segmented
            if not self.segmented:
                raise LookupError(f"The plot has not been segmented yet, please do dbscan_segment() first")
        else:
            split_in = pcd_dict

        split_out = {}
        for k in split_in.keys():
            if k == -1:
                continue
            characters = np.empty((0, 2))
            for pcd_seg in split_in[k]:
                char = np.asarray([len(pcd_seg.points) ** 0.5, calculate_xyz_volume(pcd_seg)])
                characters = np.vstack([characters, char])
            print(f'[Pnt][Plot][KMeans] class {k} Cluster Data Prepared')

            # cluster by (points number, and volumn) to remove noise segmenation
            km = KMeans(n_clusters=2)
            km.fit(characters)

            class0 = characters[km.labels_ == 0, :]
            class1 = characters[km.labels_ == 1, :]

            # find the class label with largest point clouds (plants)
            if class0.mean(axis=0)[0] > class1.mean(axis=0)[0]:
                plant_id = np.where(km.labels_ == 0)[0].tolist()
            else:
                plant_id = np.where(km.labels_ == 1)[0].tolist()

            split_out[k] = [split_in[k][pid] for pid in plant_id]

        self.pcd_segmented = split_out
        return split_out

    def sort_order(self, name_by='x', ascending=True, pcd_dict=None):
        if pcd_dict is None:
            reset_in = self.pcd_segmented
            if not self.segmented:
                raise LookupError(f"The plot has not been segmented yet, please do dbscan_segment() first")
        else:
            reset_in = pcd_dict

        reset_out = {}
        for k in reset_in.keys():
            if k == -1:
                continue
            seg_id_list = []
            x_list = []
            y_list = []
            for i, pcd in enumerate(reset_in[k]):
                x = pcd.get_center()[0]
                y = pcd.get_center()[1]
                seg_id_list.append(i)
                x_list.append(x)
                y_list.append(y)
            order_df = pd.DataFrame(dict(seg_id=seg_id_list, x=x_list, y=y_list))
            order_df = order_df.sort_values(by=name_by, ascending=ascending).reset_index()

            reset_out[k] = [reset_in[k][i] for i in order_df['seg_id']]

        self.pcd_segmented = reset_out
        return reset_out

    def save_segment_result(self, img_folder='.', show_id=True, pcd_dict=None):
        if pcd_dict is None:
            save_in = self.pcd_segmented
            if not self.segmented:
                raise LookupError(f"The plot has not been segmented yet, please do dbscan_segment() first")
        else:
            save_in = pcd_dict

        for k in save_in.keys():
            if k == -1:
                continue
            # save ply files
            pcd_id = []
            for i, pcd in enumerate(save_in[k]):
                pcd_id.append(i)
                file_name = f'class[{k}]-plant{i}'
                file_path = os.path.join(self.out_folder, f'{file_name}.ply')
                if self.write_ply:
                    o3d.io.write_point_cloud(file_path, pcd)
                    if i < 5 or i > len(save_in[k])-5:
                        print(f'[Pnt][Plot][Save_Seg] writing file "{file_path}"')

            # draw images
            if img_folder == '.':
                savepath = os.path.join(self.out_folder, f'{self.ply_name}-class[{k}].png')
            else:
                savepath = os.path.join(img_folder, f'{self.ply_name}-class[{k}].png')
            len_xyz = self.pcd_xyz.max(axis=0) - self.pcd_xyz.min(axis=0)  # calculate the size of figure
            draw_plot_seg_results(save_in[k], pcd_id,
                                  title=f'{self.ply_name}-class[{k}] ({len(save_in[k])} segments)',
                                  savepath=savepath, size=(len_xyz[0], len_xyz[1]), show_id=show_id)
            print(f'[Pnt][Plot][Save_Seg] writing image to "{savepath}"')

    def shp_segment(self, shp_dir, correct_coord=None, rename=True):
        seg_out = {}
        seg_out_name = {}
        if isinstance(shp_dir, str):   # input one shp file
            shp_seg = read_shp(shp_dir, correct_coord)
        else:
            shp_seg = read_shps(shp_dir, correct_coord=correct_coord, rename=rename)

        axis_max = self.pcd_xyz[:, 2].max()
        axis_min = self.pcd_xyz[:, 2].min()

        for k in self.pcd_classified.keys():
            if k == -1:
                continue

            seg_out[k] = []
            seg_out_name[k] = []

            print(f'[Pnt][Plot][AutoSegment][Clustering] class {k} Cluster Data Prepared')
            for plot_key in shp_seg.keys():
                # can use pcd_tools.build_cut_boundary()
                boundary = o3d.visualization.SelectionPolygonVolume()

                boundary.orthogonal_axis = "Z"
                boundary.bounding_polygon = o3d.utility.Vector3dVector(shp_seg[plot_key])
                boundary.axis_max = axis_max
                boundary.axis_min = axis_min

                roi = boundary.crop_point_cloud(self.pcd_classified[k])
                seg_out[k].append(roi)

                file_name = f'class[{k}]-{plot_key}'
                file_path = os.path.join(self.out_folder, f'{file_name}.ply')
                seg_out_name[k].append(file_name)

                if self.write_ply:
                    print(f'[Pnt][Plot][AutoSegment][Output] writing file "{file_path}"')
                    o3d.io.write_point_cloud(file_path, roi)

        self.segmented = True
        self.pcd_segmented = seg_out
        self.pcd_segmented_name = seg_out_name
        return seg_out

    def get_traits(self, container_ht=0, ground_ht='auto', savefig=True, pcd_dict=None):
        if pcd_dict is None:
            traits_in = self.pcd_segmented
            if not self.segmented:
                raise LookupError(f"The plot has not been segmented yet, please do dbscan_segment() first")
        else:
            traits_in = pcd_dict

        out_dict = {'plot': [], 'plant': [], 'kind': [], 'center.x(m)': [], 'center.y(m)': [],
                    'min_rect_width(m)': [], 'min_rect_length(m)': [], 'hover_area(m2)': [], 'PLA(cm2)': [],
                    'centroid.x(m)': [], 'centroid.y(m)': [], 'long_axis(m)': [], 'short_axis(m)': [],
                    'orient_deg2xaxis': [], 'percentile_height(m)': [], 'voxel_volume(m3)':[], 'hull3d_volume(m3)':[]}

        for k in traits_in.keys():
            number = len(traits_in[k])
            print(f'[Pnt][Plot][get_traits] total number of kind {k} is {number}')
            for i, seg in enumerate(traits_in[k]):
                plant = Plant(pcd_input=seg, indices=i, ground_pcd=self.pcd_classified[-1],
                              container_ht=container_ht, ground_ht=ground_ht)
                if savefig and self.write_ply:
                    if len(self.pcd_segmented_name) > 0:
                        file_name = self.pcd_segmented_name[k][i]
                    else:
                        file_name = f"class[{k}]-plant{i}"
                    plant.draw_3d_results(output_path=self.out_folder, file_name=file_name)
                out_dict['plot'].append(self.ply_name)
                out_dict['plant'].append(i)
                out_dict['kind'].append(k)
                out_dict['center.x(m)'].append(plant.center[0])
                out_dict['center.y(m)'].append(plant.center[1])
                out_dict['min_rect_width(m)'].append(plant.width)
                out_dict['min_rect_length(m)'].append(plant.length)
                out_dict['hover_area(m2)'].append(plant.hull_area)
                out_dict['PLA(cm2)'].append(plant.pla)
                out_dict['centroid.x(m)'].append(plant.centroid[0])
                out_dict['centroid.y(m)'].append(plant.centroid[1])
                out_dict['long_axis(m)'].append(plant.major_axis)
                out_dict['short_axis(m)'].append(plant.minor_axis)
                out_dict['orient_deg2xaxis'].append(plant.orient_degree)
                out_dict['percentile_height(m)'].append(plant.pctl_ht)
                out_dict['voxel_volume(m3)'].append(plant.voxel_volume)
                out_dict['hull3d_volume(m3)'].append(plant.hull3d_volume)

        out_pd = pd.DataFrame(out_dict)
        print(f'[Pnt][Plot][get_traits] preview of traits of first 5 of {len(out_pd)} records:')
        print(out_pd.head())

        return out_pd


class Plant(object):

    def __init__(self, pcd_input, ground_pcd, indices, cut_bg=True, container_ht=0, ground_ht='auto'):
        if isinstance(pcd_input, str):
            self.pcd = read_ply(pcd_input)
        else:
            self.pcd = pcd_input
        self.center = self.pcd.get_center()
        self.indices = indices

        self.pcd_xyz = np.asarray(self.pcd.points)
        self.pcd_rgb = np.asarray(self.pcd.colors)

        if isinstance(ground_pcd, str):   # type the ground point pcd
            self.ground_pcd = read_ply(ground_pcd)
        else:
            self.ground_pcd = ground_pcd

        # clip the background
        if cut_bg:
            self.clip_background()
            # print(f'[Pnt][Plant][clip_background] finished for No. {indices}')

        print(f'[Pnt][Plant][Traits] No. {indices} Calculating')
        # calculate the convex hull 2d
        self.plane_hull, self.hull_area = get_convex_hull(self.pcd, dim='2d')  # vertex_set (2D ndarray), m^2

        # calculate min_area_bounding_rectangle,
        # rect_res = (rot_angle, area, width, length, center_point, corner_points)
        self.rect_res = min_bounding_rect(self.plane_hull)
        self.width = self.rect_res[2]   # unit is m
        self.length = self.rect_res[3]   # unit is m

        # calculate the projected 2D image (X-Y)
        binary, px_num_per_cm, corner = pcd2binary(self.pcd)
        # calculate region props
        self.centroid, self.major_axis, self.minor_axis, self.orient_degree = self.get_region_props(binary,
                                                                                                    px_num_per_cm,
                                                                                                    corner)
        # calculate projected leaf area
        self.pla_img = binary
        self.pla = self.get_projected_leaf_area(binary, px_num_per_cm)

        # calcuate percentile height
        self.pctl_ht, self.pctl_ht_plot = self.get_percentile_height(container_ht, ground_ht)

        # voxel (todo)
        self.pcd_voxel, self.voxel_params = pcd2voxel(self.pcd)
        self.voxel_volume = self.voxel_params['voxel_number'] * (self.voxel_params['voxel_size'] ** 3)
        self.convex_hull3d, self.hull3d_volume = get_convex_hull(self.pcd, dim='3d')

    def clip_background(self):
        x_max = self.pcd_xyz[:, 0].max()
        x_min = self.pcd_xyz[:, 0].min()
        x_len = x_max - x_min
        y_max = self.pcd_xyz[:, 1].max()
        y_min = self.pcd_xyz[:, 1].min()
        y_len = y_max - y_min

        polygon = np.array([[x_min - x_len*0.1, y_min - y_len*0.1, 0],
                            [x_min - x_len*0.1, y_max + y_len*0.1, 0],
                            [x_max + x_len*0.1, y_max + y_len*0.1, 0],
                            [x_max + x_len*0.1, y_min - y_len*0.1, 0],
                            [x_min - x_len*0.1, y_min - y_len*0.1, 0]])

        ground_xyz = np.asarray(self.ground_pcd.points)
        z_max = ground_xyz[:, 2].max()
        z_min = ground_xyz[:, 2].min()

        boundary = build_cut_boundary(polygon, (z_min, z_max))

        self.ground_pcd = boundary.crop_point_cloud(self.ground_pcd)

    # -=-=-=-=-=-=-=-=-=-=-=-=
    # | traits from 2D image |
    # -=-=-=-=-=-=-=-=-=-=-=-=

    @staticmethod
    def get_region_props(binary, px_num_per_cm, corner):
        x_min, y_min = corner
        regions = regionprops(binary, coordinates='xy')
        props = regions[0]          # this is all coordinate in converted binary images

        # convert coordinate from binary images to real point cloud
        y0, x0 = props.centroid
        center = (x0 / px_num_per_cm / 100 + x_min, y0 / px_num_per_cm / 100 + y_min)

        major_axis = props.major_axis_length / px_num_per_cm / 100
        minor_axis = props.minor_axis_length / px_num_per_cm / 100

        phi = props.orientation
        angle = - phi * 180 / np.pi   # included angle with x axis, clockwise, by regionprops default

        return center, major_axis, minor_axis, angle

    @staticmethod
    def get_projected_leaf_area(binary, px_num_per_cm):
        kind, number = np.unique(binary, return_counts=True)
        # back_num = number[0]
        fore_num = number[1]

        pixel_size = (1 / px_num_per_cm) ** 2   # unit is cm2

        return fore_num * pixel_size

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    # | traits from 3D points |
    # -=-=-=-=-=-=-=-=-=-=-=-=-
    def get_percentile_height(self, container_ht=0, ground_ht='mean'):
        z = self.pcd_xyz[:, 2]
        if ground_ht == 'mean':
            ground_z = np.asarray(self.ground_pcd.points)[:, 2]
            ground_z = ground_z[ground_z < np.percentile(z, 5)]

            # calculate the ground center of Z, by mean of [per5 - per 90],
            # to avoid the effects of elevation and noises in upper part
            """
            ele = ground_z[np.logical_and(ground_z < np.percentile(ground_z, 80),
                                          ground_z > np.percentile(ground_z, 5))]
            ele = np.median(ground_z)
            """
            # ele_ht_fine = np.linspace(ground_z.min(), ground_z.max(), 1000)
            # ele_kernel = gaussian_kde(ground_z)
            # ele_hist_num = ele_kernel(ele_ht_fine)

            ele = np.median(ground_z)
            # [todo] find the largest first peaks for ground height?
        elif ground_ht=='auto':
            ground_z = np.asarray(self.ground_pcd.points)[:, 2]
            ground_z = ground_z[ground_z < np.percentile(z, 5)]
            ele = ground_z.min()
        else:
            ele = ground_ht

        print(ele, container_ht)
        plant_base = ele + container_ht

        ele_z = z[z > plant_base]
        top10percentile = np.percentile(ele_z, 90)
        plant_top = ele_z[ele_z > top10percentile].mean()

        percentile_ht = plant_top - plant_base

        plot_use = {'plant_top': plant_top, 'plant_base': plant_base,
                    'top10': top10percentile, 'ground_center': ele}

        return percentile_ht, plot_use

    def draw_3d_results(self, output_path='.', file_name=None):
        if file_name is None:
            plant_name = f'plant{self.indices}'
        else:
            plant_name = file_name

        file_name = f'{plant_name}.png'
        draw_3d_results(self, title=plant_name, savepath=f"{output_path}/{file_name}")