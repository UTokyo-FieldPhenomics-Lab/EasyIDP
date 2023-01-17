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


def draw_polygon_on_img(img_name, img_path, poly_coord, corrected_poly_coord=None, title=None, save_as=None, show=False,
                        color='red', alpha=0.5, dpi=72):
    """Plot one polygon on given image.

    Parameters
    ----------
    img_name : str
        the image file name.
    img_path : str
        the file path of image
    poly_coord : np.ndarray
        the 2D polygon pixel coordinate on the image
    corrected_poly_coord : np.ndarray, optional
        the corrected 2D polygon pixel coordiante on the image (if have), by default None
    title : str, optional
        The image title displayed on the top, by default None -> ``Projection on [img_name]``
    save_as : str, optional
        file path to save the output figure, by default None
    show : bool, optional
        whether display (in jupyter notebook) or popup (in command line) the figure, by default False
    color : str, optional
        the polygon line color, by default 'red'
    alpha : float, optional
        the polygon transparency, by default 0.5
    dpi : int, optional
        the dpi of produced figure, by default 72

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp

        >>> p4d = idp.Pix4D(
        ...     test_data.pix4d.lotus_folder,
        ...     raw_img_folder=test_data.pix4d.lotus_photos,
        ...     param_folder=test_data.pix4d.lotus_param
        ... )

        >>> plot =  np.array([   # N1E1 plot geo coordinate
        ...     [ 368020.2974959 , 3955511.61264302,      97.56272272],
        ...     [ 368022.24288365, 3955512.02973983,      97.56272272],
        ...     [ 368022.65361232, 3955510.07798313,      97.56272272],
        ...     [ 368020.69867274, 3955509.66725421,      97.56272272],
        ...     [ 368020.2974959 , 3955511.61264302,      97.56272272]
        ... ])

    Then do backward projection, find the previous ROI positions on the raw images.

    .. code-block:: python

        >>> out_dict = p4d.back2raw_crs(plot, distort_correct=True)
        >>> out_dict["DJI_0177.JPG"]
        array([[ 137.10982937, 2359.55887614],
               [ 133.56116243, 2107.13954299],
               [ 384.767746  , 2097.05639105],
               [ 388.10993307, 2350.41225998],
               [ 137.10982937, 2359.55887614]])

    The using this function to check one polygon on raw images

    .. code-block:: python

        >>> img_name = "DJI_0198.JPG"
        >>> photo = p4d.photos[img_name]

        >>> idp.visualize.draw_polygon_on_img(
        ...     img_name, photo.path, out_dict[img_name],  
        ...     save_as="p4d_back2raw_single_view.png")

    If will get the following figure:

    .. image:: ../../_static/images/visualize/p4d_back2raw_single_view.png
        :alt: p4d_back2raw_single_view.png'
        :scale: 100

    Add an corrected polygon (here manual shifting 10 pixels just as example), and change the color and alpha info:

    .. code-block:: python

        >>> corrected_poly = out_dict[img_name] + np.array([10,10])

        >>> idp.visualize.draw_polygon_on_img(
        ...     img_name, photo.path, out_dict[img_name], corrected_poly,
        ...     save_as="p4d_back2raw_single_view2.png", 
        ...     color='blue', alpha=0.3)

    If will get the following figure:

    .. image:: ../../_static/images/visualize/p4d_back2raw_single_view2.png
        :alt: p4d_back2raw_single_view2.png'
        :scale: 100

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
    img_array = plt.imread(img_path)
    ax[0].imshow(img_array)

    if corrected_poly_coord is None:
        polygon = pts.Polygon(poly_coord, True)
    else:
        polygon = pts.Polygon(corrected_poly_coord, True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=color)

    ax[0].add_collection(p)

    if title is None:
        plt.title(f"Reprojection on [{img_name}]")
    else:
        plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')

    ax[1].imshow(img_array)

    if corrected_poly_coord is None:
        x_min, y_min = poly_coord[:,0:2].min(axis=0)
        x_max, y_max = poly_coord[:,0:2].max(axis=0)

        ax[1].plot(poly_coord[:, 0], poly_coord[:, 1], color=color, linestyle='-')
    else:
        x_min, y_min = np.vstack([poly_coord, corrected_poly_coord])[:,0:2].min(axis=0)
        x_max, y_max = np.vstack([poly_coord, corrected_poly_coord])[:,0:2].max(axis=0)

        l1, = ax[1].plot(poly_coord[:, 0], poly_coord[:, 1], color=color, linestyle='--')
        l2, = ax[1].plot(corrected_poly_coord[:, 0], corrected_poly_coord[:, 1], color=color, linestyle='-')
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