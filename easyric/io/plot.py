import os
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection

def draw_polygon_on_img(param, img_name, img_coord, title=None, file_name=None, show=False, color='red', alpha=0.5, dpi=72):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
    ax.imshow(imread(param.img[img_name].path))

    polygon = pts.Polygon(img_coord, True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=color)

    ax.add_collection(p)
    if title is None:
        ax.set_title(f"Reprojection on [{img_name}]")
    else:
        ax.set_title(title)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)

    if show:
        plt.show()

    plt.close()

def draw_polygon_on_imgs(param, img_names, img_coords, out_folder, coord_prefix, color='red', alpha=0.5, dpi=72):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        print(f'[IO][Plot] making folder {os.path.abspath(out_folder)}')

    for img_n, img_c in zip(img_names, img_coords):
        title = f"Reprojection {coord_prefix} on [{img_n}]"
        file_name = f"{coord_prefix}_{img_n}.png"
        draw_polygon_on_img(param, img_n, img_c, title=title, file_name=out_folder + '/' + file_name,
                            color=color, alpha=alpha, dpi=dpi)