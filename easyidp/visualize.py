import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection
from tqdm import tqdm
import warnings


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

    It will get the following figure:

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

    It will get the following figure:

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


def draw_backward_one_roi(proj, result_dict, buffer=40, title=None, save_as=None, show=False, color='red', alpha=0.5, dpi=72):
    """Plot one ROI results on all available images.

    Parameters
    ----------
    proj : easyidp.Recons
        The 3D reconstruction project object
    result_dict : dict
        | The dictionary of one ROI backward to all images.
        | e.g. ``{"IMG_2345": np.array([...]), "IMG_2346": np.array([])}``
    buffer : int, optional
        The pixel buffer number around the backward ROI results, by default 40
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

    Data prepare:

    .. code-block:: python

        >>> import easyidp as idp
        >>> lotus = idp.data.Lotus()

        >>> roi = idp.ROI(lotus.shp, name_field='plot_id')
        >>> roi.get_z_from_dsm(lotus.metashape.dsm)

        >>> ms = idp.Metashape(lotus.metashape.project, chunk_id=0)
        >>> img_dict_ms = roi.back2raw(ms)

    Then use this code to show the results of ROI [N1W1]:

    .. code-block:: python

        >>> idp.visualize.draw_backward_one_roi(ms, img_dict_ms['N1W1'], save_as="draw_backward_one_roi.png")

    It will get the following figure:

    .. image:: ../../_static/images/visualize/draw_backward_one_roi.png
        :alt: draw_backward_one_roi.png'

    """
    title_list = ['ROI Positions on Original Images', 'Enlarge Detail View']
    if title is not None:
        if isinstance(title, list) and len(title) == 2:
            title_list = title
        else:
            warnings.warn(f"Expected title like ['title1', 'title2'], not given '{title}', using default title instead")

    img_num = len(result_dict) + 1
    grid_w = np.ceil(np.sqrt(img_num)).astype(int)

    if img_num % grid_w == 0:  # no need a new line
        grid_h = (img_num // grid_w).astype(int)
    else:
        grid_h = (img_num // grid_w + 1).astype(int)

    ratio = grid_h / grid_w

    # grid_w * 3 -> the recommended size of image
    # grid_w * 3 * 2 -> the relative width of image, doubed due to two figures connected together
    # griw_w * 3 * ratio -> the relative height of image
    fig, ax = plt.subplots(ncols=grid_w*2, nrows=grid_h, figsize=(grid_w*3*2, grid_w*3*ratio), dpi=dpi)

    tbar = tqdm(result_dict, desc=f"Reading image files for plotting")
    for i, example_img in enumerate(tbar):
        img_np = plt.imread(proj.photos[example_img].path)
        img_coord = result_dict[example_img]
        im_xmin, im_ymin = img_coord.min(axis=0)
        im_xmax, im_ymax = img_coord.max(axis=0)

        img_id = i
        img_h = img_id // grid_w
        img_w = img_id % grid_w

        # print(f"img_id={img_id}; img_h={img_h}; img_w={img_w}", end='\r')

        polygon = pts.Polygon(img_coord, True)
        p = PatchCollection([polygon], alpha=alpha, facecolors=color)

        # draw roi on full image
        ax[img_h, img_w].imshow(img_np)
        # ax[img_h, img_w].plot(*img_coord.T, '--', c=color)
        ax[img_h, img_w].add_collection(p)
        ax[img_h, img_w].set_xlabel(example_img, size='x-large')
        ax[img_h, img_w].invert_yaxis()

        # draw roi on zoomed image
        ax[img_h, img_w + grid_w].imshow(img_np)
        ax[img_h, img_w + grid_w].plot(*img_coord.T, '--', c=color)
        ax[img_h, img_w + grid_w].set_xlim(im_xmin - buffer, im_xmax + buffer)
        ax[img_h, img_w + grid_w].set_ylim(im_ymin - buffer, im_ymax + buffer)
        ax[img_h, img_w + grid_w].set_xlabel(example_img, size='x-large')
        ax[img_h, img_w + grid_w].invert_yaxis()

    print(f"Image data loaded, drawing figures, this may cost a few seconds...")

    for empty_idx in range(img_w+1, grid_w):
        ax[img_h, empty_idx].axis('off')
        ax[img_h, empty_idx + grid_w].axis('off')

    plt.tight_layout()

    # make space for suptitle
    # y = -0.1/x + 0.99, in range 0.9-0.98
    plt.subplots_adjust(top= -0.1 / grid_h + 0.99)

    # add subfigure title
    plt.text(.25, 0.99, 
        title_list[0], 
        transform=fig.transFigure, 
        horizontalalignment='center', 
        verticalalignment='top', 
        size='xx-large')
    plt.text(.75, 0.99, 
        title_list[1], 
        transform=fig.transFigure, 
        horizontalalignment='center', 
        verticalalignment='top', 
        size='xx-large')

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()

    plt.clf()
    plt.close(fig)
    del fig, ax, img_np