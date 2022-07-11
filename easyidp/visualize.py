import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection


def _view_poly2mask(poly, mask, pix_all, pix_in):
    """_summary_

    Parameters
    ----------
    poly : np.ndarray
        (horizontal, vertical) = (width, height)
        !!! it is reversed with numpy index order !!!
    mask : np.ndarray
        the binary mask
    pix_all : np.ndarray
        the coordinate of all pixels
        (horizontal, vertical) = (width, height)
        !!! it is reversed with numpy index order !!!
    pix_in : np.ndarray
        the coordinate of all "in-polygon" pixels
        (horizontal, vertical) = (width, height)
        !!! it is reversed with numpy index order !!!
    """
    h, w = mask.shape

    fig, ax = plt.subplots(1,1, figsize=(7, 7 * h/w))

    # draw figures
    # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
    ax.imshow(mask, extent=[0, w, 0, h], origin="lower", cmap="Pastel1_r")

    # -------------
    # draw scatters
    # -------------
    ax.scatter(*np.array(pix_all).T, c='k')
    ax.plot(*poly.T)
    ax.scatter(*np.array(pix_in).T, c='r')
    ax.axis('equal')

    #plt.grid()

    ax.set_xlim(0,w)
    ax.set_ylim(0,h)

    ax.set_xlabel("Y")
    ax.set_ylabel("X")

    plt.gca().invert_yaxis()

    plt.show()


def draw_polygon_on_img(img_name, img_path, img_coord, img_correct=None, title=None, save_as=None, show=False,
                        color='red', alpha=0.5, dpi=72):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
    img_array = plt.imread(img_path)
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
    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()

    plt.clf()
    plt.close(fig)
    del fig, ax, img_array