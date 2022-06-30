import numpy as np
import matplotlib.pyplot as plt


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