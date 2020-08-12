import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection


def draw_polygon_on_img(param, img_name, img_coord, img_correct=None, title=None, file_name=None, show=False,
                        color='red', alpha=0.5, dpi=72):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
    img_array = imread(param.img[img_name].path)
    ax[0].imshow(img_array)

    if img_correct is None:
        polygon = pts.Polygon(img_coord, True)
    else:
        polygon = pts.Polygon(img_correct, True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=color)

    ax[0].add_collection(p)

    if title is None:
        plt.title(f"Reprojection on [{img_name}]")
    else:
        plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')

    ax[1].imshow(img_array)

    if img_correct is None:
        x_min, y_min = img_coord[:,0:2].min(axis=0)
        x_max, y_max = img_coord[:,0:2].max(axis=0)

        ax[1].plot(img_coord[:, 0], img_coord[:, 1], color=color, linestyle='-')
    else:
        x_min, y_min = np.vstack([img_coord, img_correct])[:,0:2].min(axis=0)
        x_max, y_max = np.vstack([img_coord, img_correct])[:,0:2].max(axis=0)

        l1, = ax[1].plot(img_coord[:, 0], img_coord[:, 1], color=color, linestyle='--')
        l2, = ax[1].plot(img_correct[:, 0], img_correct[:, 1], color=color, linestyle='-')
        ax[1].legend((l1, l2), ('Original Projection', 'Corrected Position'), loc='lower center')

    x_shift = (x_max - x_min) * 0.1
    y_shift = (y_max - y_min) * 0.1

    ax[1].set_xlim([x_min - x_shift, x_max + x_shift])
    ax[1].set_ylim([y_min - y_shift, y_max + y_shift])
    ax[1].invert_yaxis()

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)

    if show:
        plt.show()

    plt.clf()
    plt.close(fig)
    del fig, ax, img_array


def draw_polygon_on_imgs(param, img_coord_dict, out_folder, coord_prefix, img_correct_dict=None, color='red', alpha=0.5, dpi=72):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f'[IO][Plot] making folder {os.path.abspath(out_folder)}')

    i = 1
    for img_n, img_c in img_coord_dict.items():
        print(f'[IO][Plot] Drawing {img_n}|{i} of {len(img_coord_dict.keys())}', end='\r')
        title = f"Reprojection {coord_prefix} on [{img_n}]"
        file_name = f"{coord_prefix}_{img_n}.png"
        if img_correct_dict is None:
            img_correct = None
        else:
            img_correct = img_correct_dict[img_n]
        draw_polygon_on_img(param, img_n, img_c, title=title, file_name=out_folder + '/' + file_name,
                            img_correct=img_correct, color=color, alpha=alpha, dpi=dpi)
        i += 1