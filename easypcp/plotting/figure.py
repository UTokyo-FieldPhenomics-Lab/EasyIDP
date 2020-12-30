import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.stats import gaussian_kde

from easypcp.pcd_tools import get_convex_hull

def draw_plot_seg_results(pcd_seg_list, selected_id_list, title, savepath, show_id=True, size=(9, 6), dpi=300):
    """
    :param pcd_seg_list: [geometry::PointCloud with 35 points., geometry::PointCloud with 76 points., ...]
    :param selected_id_list: [5, 14, 8, 12, 21, 22, 23]
    :param title: str
    :param savename: str
    :return:
    """

    # =-=-=-=-=-=-=-=-=-=-=-=-=-
    # |  variables calculation |
    # -=-=-=-=-=-=-=-=-=-=-=-=-=

    # prepare for colored scatters
    scatter_xyrgba = np.zeros((0, 6))
    for i in range(len(pcd_seg_list)):
        if i in selected_id_list:
            alpha = 0.7
        else:
            alpha = 0.2
        pcd_temp = copy(pcd_seg_list[i])
        points_num = len(pcd_temp.points)
        if points_num > 500:
            k = int(points_num / 500)
            pcd_temp = pcd_temp.uniform_down_sample(k)

        pcd_temp.paint_uniform_color(np.random.rand(3))
        xyz = np.asarray(pcd_temp.points)
        rgb = np.asarray(pcd_temp.colors)
        np_alpha = np.ones((len(xyz), 1)) * alpha
        xyrgba = np.hstack([xyz[:, 0:2], rgb, np_alpha])
        scatter_xyrgba = np.vstack([scatter_xyrgba, xyrgba])

    # prepare for kept segments
    convex_container = []
    center_container = []
    text_container = []
    for i, sid in enumerate(selected_id_list):
        plane_hull, _ = get_convex_hull(pcd_seg_list[sid], dim='2d')
        convex_hull = np.vstack([plane_hull, plane_hull[0, :]])

        convex_container.append(convex_hull)
        center_container.append(pcd_seg_list[sid].get_center())
        text_container.append(str(i))

    w, h = size
    max_len = max(w, h)
    ratio = 10 / max_len
    # -=-=-=-=-=-=-=
    # | draw plots |
    # -=-=-=-=-=-=-=
    fig, ax = plt.subplots(figsize=(w * ratio, h * ratio), dpi=dpi)

    ax.scatter(x=scatter_xyrgba[:, 0], y=scatter_xyrgba[:, 1], c=scatter_xyrgba[:, 2:], marker='.', linewidth=0)
    for i in range(len(text_container)):
        center = center_container[i]
        convex = convex_container[i]
        text = text_container[i]
        ax.plot(convex[:, 0], convex[:, 1], 'r--')
        if show_id:
            ax.text(x=center[0], y=center[1], s=text, ha='center', va='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, size=10)

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.clf()
    plt.close(fig)
    del fig, ax

def draw_3d_results(plant, title, savepath, dpi=300):
    # =-=-=-=-=-=-=-=-=-=-=-=
    # |  traits calculation |
    # -=-=-=-=-=-=-=-=-=-=-=-
    x = plant.pcd_xyz[:, 0]
    y = plant.pcd_xyz[:, 1]
    z = plant.pcd_xyz[:, 2]

    convex_hull = np.vstack([plant.plane_hull, plant.plane_hull[0, :]])

    corner = plant.rect_res[5]
    rect_corner = np.vstack([corner, corner[0, :]])

    x0 = plant.centroid[0]
    y0 = plant.centroid[1]
    phi = plant.orient_degree

    maj_x = np.cos(np.deg2rad(phi)) * 0.5 * plant.major_axis
    maj_y = np.sin(np.deg2rad(phi)) * 0.5 * plant.major_axis
    min_x = np.sin(np.deg2rad(phi)) * 0.5 * plant.minor_axis
    min_y = np.cos(np.deg2rad(phi)) * 0.5 * plant.minor_axis

    downpcd = plant.pcd.voxel_down_sample(voxel_size=plant.voxel_size)
    downpcd_xyz = np.asarray(downpcd.points)
    down_color = np.asarray(downpcd.colors)
    down_x = downpcd_xyz[:, 0]
    down_y = downpcd_xyz[:, 1]
    down_z = downpcd_xyz[:, 2]

    ground_pcd = plant.ground_pcd.voxel_down_sample(voxel_size=plant.voxel_size * 5)
    ground_pcd_xyz = np.asarray(ground_pcd.points)
    ground_x = ground_pcd_xyz[:, 0]
    ground_y = ground_pcd_xyz[:, 1]
    ground_z = ground_pcd_xyz[:, 2]
    mask = ground_z < np.percentile(z, 5)
    ground_x = ground_x[mask]
    ground_y = ground_y[mask]
    ground_z = ground_z[mask]

    x_axis_max = round(max(x.max(), rect_corner[:, 0].max(), ground_x.max()), 2)
    x_axis_min = round(min(x.min(), rect_corner[:, 0].min(), ground_x.min()), 2)
    y_axis_max = round(max(y.max(), rect_corner[:, 1].max(), ground_y.max()), 2)
    y_axis_min = round(min(y.min(), rect_corner[:, 1].min(), ground_y.min()), 2)
    z_axis_max = round(max(z.max(), ground_z.max()), 2)
    z_axis_min = round(min(z.min(), ground_z.min()), 2)

    pctl_ht, pctl_ht_plot = plant.pctl_ht, plant.pctl_ht_plot

    # perpare for gasussian kde histgramme
    ele_ht = z[z > pctl_ht_plot['plant_base']]
    ele_ht_fine = np.linspace(ele_ht.min(), ele_ht.max(), 1000)
    ele_kernel = gaussian_kde(ele_ht)
    ele_hist_num = ele_kernel(ele_ht_fine)

    kde_max = ele_hist_num.max()
    bar_len = (x_axis_max - x_axis_min) * 0.1
    ele_hist_num_plot = ele_hist_num * (bar_len / kde_max) - (0 - x_axis_min)

    # -=-=-=-=-=-=-=
    # | draw plots |
    # -=-=-=-=-=-=-=
    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    ax = Axes3D(fig, elev=25, azim=135, rect=(0, 0.07, 1, 0.95))

    # zorder 2,3
    # plot the ground on Y-Z
    ax.scatter(ground_y, ground_z, zs=x_axis_max, zdir='x',
               color='C4', linewidth=0, marker='.', alpha=0.3, label='ground points')
    # plot the ground on X-Z
    ax.scatter(ground_x, ground_z, zs=y_axis_min, zdir='y',
               color='C4', linewidth=0, marker='.', alpha=0.3)

    # zorder 4,5,6
    # plot the shade on X-Y
    ax.scatter(down_x, down_y, zs=z_axis_min, zdir='z',
               color='C7', linewidth=0, marker='.', alpha=0.3)
    # plot the shade on Y-Z
    ax.scatter(down_y, down_z, zs=x_axis_max, zdir='x',
               color='C7', linewidth=0, marker='.', alpha=0.3)
    # plot the shade on X-Z
    ax.scatter(down_x, down_z, zs=y_axis_min, zdir='y',
               color='C7', linewidth=0, marker='.', alpha=0.3)

    # zorder 7
    # plot convex hull convet
    ax.scatter(convex_hull[:, 0], convex_hull[:, 1], color='C3', linewidth=0,
               zdir='z', zs=z_axis_min, alpha=1,
               label='convex vertex')

    # zorder 8
    # plot 3d point clouds
    ax.scatter(down_x, down_y, down_z, marker='.', color=down_color, alpha=0.7, linewidth=0, zorder=25)

    # plot convex hull
    ax.plot(convex_hull[:, 0], convex_hull[:, 1], 'C0--', alpha=0.5,
            zdir='z', zs=z_axis_min,
            label='convex hull')
    ax.plot(rect_corner[:, 0], rect_corner[:, 1], color='C1',
            alpha=0.5, zdir='z', zs=z_axis_min,
            label='min area rectangle')

    # plot the ellipse of region props
    ell = Ellipse((x0, y0), plant.major_axis, plant.minor_axis, phi,
                  alpha=0.2, zorder=0, label='region props')
    ax.add_patch(ell)
    art3d.pathpatch_2d_to_3d(ell, z=z_axis_min, zdir="z")
    # plot axis of region props
    ax.plot((x0 - maj_x, x0 + maj_x), (y0 - maj_y, y0 + maj_y), '-k',
            linewidth=2.5, zdir='z', zs=z_axis_min, zorder=7.5,
            label='ellipse axis')
    ax.plot((x0 + min_x, x0 - min_x), (y0 - min_y, y0 + min_y), '-k',
            linewidth=2.5, zdir='z', zs=z_axis_min, zorder=7.5)

    # plot height lines
    ## container height != 0
    if pctl_ht_plot['ground_center'] != pctl_ht_plot['plant_base']:
        container_ht = round((pctl_ht_plot['plant_base'] - pctl_ht_plot['ground_center']) * 100, 1)  # unit is cm
        # draw gorund height
        ax.plot((y_axis_min, y_axis_max),
                (pctl_ht_plot['ground_center'], pctl_ht_plot['ground_center']), '--',
                zs=x_axis_max, zdir='x', color='C1', label='ground height', zorder=7.5)
        ax.plot((x_axis_min, x_axis_max),
                (pctl_ht_plot['ground_center'], pctl_ht_plot['ground_center']), '--',
                zs=y_axis_min, zdir='y', color='C1', zorder=7.5)
        ax.text(x_axis_max, y_axis_max, (pctl_ht_plot['ground_center'] + pctl_ht_plot['plant_base']) / 2,
                f'Container {container_ht} cm', zdir="y", zorder=10)

    ## draw plant base
    ax.plot((y_axis_min, y_axis_max),
            (pctl_ht_plot['plant_base'], pctl_ht_plot['plant_base']), '--',
            zs=x_axis_max, zdir='x', color='C2', zorder=7.5,
            label='plant base')
    ax.plot((x_axis_min, x_axis_max),
            (pctl_ht_plot['plant_base'], pctl_ht_plot['plant_base']), '--',
            zs=y_axis_min, zdir='y', color='C2', zorder=7.5)
    ## draw plant top
    ax.plot((y_axis_min, y_axis_max),
            (pctl_ht_plot['plant_top'], pctl_ht_plot['plant_top']), '--',
            zs=x_axis_max, zdir='x', color='C8', zorder=7.5,
            label='plant top')
    ax.plot((x_axis_min, x_axis_max),
            (pctl_ht_plot['plant_top'], pctl_ht_plot['plant_top']), '--',
            zs=y_axis_min, zdir='y', color='C8', zorder=7.5)

    ## draw kde
    ax.plot(ele_hist_num_plot, ele_ht_fine, zs=y_axis_min, zdir='y',
            color='C2', zorder=8, label='plant density')

    ## draw texts
    ax.text(x_axis_max, y_axis_max, (pctl_ht_plot['plant_base'] + pctl_ht_plot['plant_top']) / 2,
            f'Plant {round(pctl_ht * 100, 1)} cm', zdir="y", zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)
    ax.set_zlim(z_axis_min, z_axis_max)

    ax.set_title(title, size=20)

    plt.legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.07))
    plt.savefig(savepath)
    plt.clf()
    plt.close(fig)
    del fig, ax