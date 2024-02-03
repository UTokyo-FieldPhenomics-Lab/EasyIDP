.. EasyIDP documentation master file, created by
   sphinx-quickstart on Sun Jun  5 11:43:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
Welcome to EasyIDP's documentation!
===================================

.. image:: _static/images/header_v2.0.png
   :alt: header_v2.0.png

EasyIDP (Easy Intermediate Data Processor) is a handy tool for dealing with region of interest (ROI) on the image reconstruction (Metashape & Pix4D) outputs, mainly in agriculture applications. It provides the following functions:

- Backward Projection ROI to original images.
- Clip ROI on GeoTiff Maps (DOM & DSM) and Point Cloud.
- Save cropped results to corresponding files

.. _Sphinx: http://sphinx-doc.org

.. image:: https://img.shields.io/pypi/pyversions/easyidp?style=plastic
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic
   :alt: GitHub

.. image:: https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic
   :alt: GitHub top language

.. image:: https://img.shields.io/github/downloads/UTokyo-FieldPhenomics-Lab/EasyIDP/total?label=github%20downloads&style=plastic
   :alt: GitHub Downloads

.. image:: https://img.shields.io/pypi/dm/easyidp?color=%233775A9&label=pypi%20downloads&style=plastic
   :alt: Pypi downloads
   :target: https://pypi.org/project/easyidp/


.. admonition:: other languages
   
   This is a multi-language document, you can change document languages here or at the bottom left corner.

   .. hlist::
      :columns: 3

      * `English <https://easyidp.readthedocs.io/en/latest/>`_
      * `中文 <https://easyidp.readthedocs.io/zh_CN/latest/>`_
      * `日本語(翻訳募集) <https://easyidp.readthedocs.io/ja/latest/>`_



.. note::

    In the EasyIDP, we use the (horizontal, vertical, dim) order as the coords order. When it applies to the GIS coordintes, is (longitude, latitude, altitude)


Examples
========

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Backgrounds

   backgrounds/virtualenv
   backgrounds/roi_marking
   backgrounds/geotiff_transparency

.. nbgallery::
   :glob:
   :caption: Examples

   jupyter/load_roi
   jupyter/crop_outputs
   jupyter/backward_projection
   jupyter/get_z_from_dsm
   jupyter/forward_projection

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Python API

   python_api/index
   python_api/data
   python_api/pointcloud
   python_api/geotiff
   python_api/cvtools
   python_api/roi
   python_api/shp
   python_api/json
   python_api/geotools
   python_api/reconstruct
   python_api/pix4d
   python_api/metashape
   python_api/visualize

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contribute


Quick Start
===========


You can install the packages by PyPi:

.. code-block:: bash

    pip install easyidp


.. note::

    If you meet bugs in the pypi version, please consider using the latest source code. The tutorial can be found here: :ref:`using-from-source-code`.


And import the packages in your python code:

.. code-block:: python

    import easyidp as idp


.. caution::

    Before doing the following example, please understand the basic pipeline for image 3D reconstruction by Pix4D or Metashape. And know how to export the DOM, DSM (\*.tiff), and Point cloud (\*.ply). Also require some basic knowledge about GIS shapefile format (\*.shp).



1. Read ROI
------------


.. code-block:: python

    roi = idp.ROI("xxxx.shp")  # lon and lat 2D info
    
    # get z values from DSM
    roi.get_z_from_dsm("xxxx_dsm.tiff")  # add height 3D info


The 2D roi can be used to crop the DOM, DSM, and point cloud ( `2. Crop by ROI`_ ). While the 3D roi can be used for Backward projection ( `4. Backward Projection`_ )

.. caution::

    It is highly recommended to ensure the shapefile and geotiff share the same coordinate reference systems (CRS), the built-in conversion algorithm in EasyIDP may suffer accuracy loss.

    - It is recommended to use Coordinate reference systems for "UTM" (WGS 84 / UTM grid system), the unit for x and y are in meters and have been tested by EasyIDP developers.

    - The traditional (longitude, latitude) coordinates like ``epsg::4326`` also supported, but not recommended if you need calculate "distences" hence its unit is degree.

    - The local country commonly used coordinates like BJZ54 (北京54), CGCS2000 (2000国家大地坐标系), JDG2011 (日本測地系2011), and etc., have not been tested and hard to guarantee workable in EasyIDP. Please convert to UTM by GIS software if you meet any problem.

    The acceptable ROI types are only polygons (grids are a special type of polygon), and the size of each polygon should be fittable into the raw images (around the 1/4 size of one raw image should be the best).

    .. figure:: _static/images/roi_types.png
        :alt: ROI types

        The fourth one may too large to be fitted into each raw image, recommend to make smaller polygons.

2. Crop by ROI
--------------

Read the DOM and DSM Geotiff Maps

.. code-block:: python
    
    dom = idp.GeoTiff("xxx_dom.tif")
    dsm = idp.GeoTiff("xxx_dsm.tif")
  
Read point cloud data

.. code-block:: python

    ply = idp.PointCloud("xxx_pcd.ply")

  
crop the region of interest from ROI:

.. code-block:: python

    dom_parts = roi.crop(dom)
    dsm_parts = roi.crop(dsm)
    pcd_parts = roi.crop(ply)

If you want to save these crops to given folder:

.. code-block:: python

    dom_parts = roi.crop(dom, save_folder="./crop_dom")
    dsm_parts = roi.crop(dsm, save_folder="./crop_dsm")
    pcd_parts = roi.crop(ply, save_folder="./crop_pcd")

  
3. Read Reconstruction projects
-------------------------------

You can add the reconstructed plot individually or by batch adding

.. Add one reconstructed plot
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab:: Metashape

    The Metashape projects naturally support using different chunks for different plots in one project file (\*.psx), so the ``chunk_id`` is required to specify which plot are processing.

    .. code-block:: python

        ms = idp.Metashape("xxxx.psx", chunk_id=0)

    .. caution::

        Though only the ``xxxx.psx`` is given here, the folder ``xxxx.files`` generated by Metashape is more important for EasyIDP. Please put them into the same directory.

.. tab:: Pix4D

    Currently, the EasyIDP has not support parse the meta info that records the relative path to the raw image folder, so please manual specify the ``raw_img_folder`` if you need the backward projection.


    .. code-block:: python

        p4d = idp.Pix4D(project_path="xxxx.p4d", 
                        raw_img_folder="path/to/folders/with/raw/photos/",
                        # optional, in case you changed the pix4d project folder
                        param_folder="path/to/pix4d/parameter/folders")

    .. caution::

        Though only the ``xxxx.p4d`` is given here, the folder ``xxxx`` generated by Pix4D is more important for EasyIDP. Please put them into the same directory and not recommend the change the inner folder structure



.. Batch pool for multi-plots
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. .. todo::

..     This feather has not supported yet.


..     Add the reconstruction projects to processing pools (different flight time for the same field):
    
..     .. code-block:: python

..         proj = idp.ProjectPool()
..         proj.add_pix4d(["date1.p4d", "date2.p4d", ...])
..         proj.add_metashape(["date1.psx", "date2.psx", ...])


..     Then you can specify each chunk by:

..     .. code-block:: python

..         p1 = proj[0]
..         # or
..         p1 = proj["chunk_or_project_name"]


4. Backward Projection
----------------------
  
.. code-block:: python
    
    img_dict = roi.back2raw(chunk1)

  
Then check the results:

.. code-block:: python

    # find the raw image name list
    >>> img_dict.keys()   
    dict_keys(['DJI_0177', 'DJI_0178', 'DJI_0179', 'DJI_0180', ... ]

    # the roi pixel coordinate on that image
    >>> img_dict['DJI_0177'] 
    array([[ 779,  902],
           [1043,  846],
           [1099, 1110],
           [ 834, 1166],
           [ 779,  902]])


Save backward projected images

.. tab:: Metashape

    .. code-block:: python

        img_dict = roi.back2raw(ms, save_folder="folder/to/put/results/")


.. tab:: Pix4D

    .. code-block:: python

        img_dict = roi.back2raw(p4d, save_folder="folder/to/put/results/")


5. Forward Projection
---------------------

This function support bring results and image itself from corresponding raw images to geotiff.

.. caution:: The following features have not been supported.
        
Please using the following code to forward project from raw img to DOM (``raw forward dom`` -> ``raw4dom``):

This feather has not supported yet.

.. code-block:: python

    imarray_dict = roi.raw4dom(ms)

This is a dict contains the image ndarray of each ROI as keys, which projecting the part of raw image onto DOM.

.. code-block:: python

    >>> geotif = imarray_dict['roi_111']
    >>> geotif
    <class 'easyidp.geotiff.GeoTiff'>

You can also do the forward projecting with detected results, the polygon are in the geo coordinate. Before doing that, please prepare the detected results (by detection or segmentation, polygons in raw image pixel coordinates).


.. code-block:: python

    result_on_raw = idp.roi.load_results_on_raw_img(r"C:/path/to/detected/results/folder/IMG_2345.txt")

Then forward projecting by giving both to the function:

.. code-block:: python

    >>> imarray_dict, coord_dict = roi.raw4dom(ms, result_on_raw)
    >>> coord_dict['roi_111']
    {0: array([[139.54052962,  35.73475194],
               ...,
               [139.54052962,  35.73475194]]), 
     ...,
     1: array([[139.54053488,  35.73473289],
               ...,
               [139.54053488,  35.73473289]]),


Save single geotiff of one ROI:

.. code-block:: python

    >>> geotif = imarray_dict['roi_111']
    >>> geotif.save(r"/file/path/to/save/geotif_roi_111.tif")

Batch save single geotiff of each ROI:

.. code-block:: python

    >>> imarray_dict = roi.raw4dom(ms, save_folder="/path/to/save/folder")



References
==========

Publication
-----------

Please cite this paper if this project helps you: 

.. code-block:: latex

    @Article{wang_easyidp_2021,
    AUTHOR = {Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei},
    TITLE = {EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping},
    JOURNAL = {Remote Sensing},
    VOLUME = {13},
    YEAR = {2021},
    NUMBER = {13},
    ARTICLE-NUMBER = {2622},
    URL = {https://www.mdpi.com/2072-4292/13/13/2622},
    ISSN = {2072-4292},
    DOI = {10.3390/rs13132622}
    }

Site packages
--------------

We also thanks the benefits from the following open source projects:

* package main (**for users**)

  * numpy: `https://numpy.org/ <https://numpy.org/>`_ 
  * matplotlib: `https://matplotlib.org/ <https://matplotlib.org/>`_ 
  * scikit-image: `https://github.com/scikit-image/scikit-image <https://github.com/scikit-image/scikit-image>`_ 
  * pyproj: `https://github.com/pyproj4/pyproj <https://github.com/pyproj4/pyproj>`_ 
  * tifffile: `https://github.com/cgohlke/tifffile <https://github.com/cgohlke/tifffile>`_ 
  * imagecodecs: `https://github.com/cgohlke/imagecodecs <https://github.com/cgohlke/imagecodecs>`_ 
  * shapely: `https://github.com/shapely/shapely <https://github.com/shapely/shapely>`_ 
  * laspy/lasrs/lasio: `https://github.com/laspy/laspy <https://github.com/laspy/laspy>`_ 
  * geojson: `https://github.com/jazzband/geojson <https://github.com/jazzband/geojson>`_ 
  * plyfile: `https://github.com/dranjan/python-plyfile <https://github.com/dranjan/python-plyfile>`_ 
  * pyshp: `https://github.com/GeospatialPython/pyshp <https://github.com/GeospatialPython/pyshp>`_ 
  * tabulate: `https://github.com/astanin/python-tabulate <https://github.com/astanin/python-tabulate>`_ 
  * tqdm: `https://github.com/tqdm/tqdm <https://github.com/tqdm/tqdm>`_ 
  * gdown: `https://github.com/wkentaro/gdown <https://github.com/wkentaro/gdown>`_ 

* package documentation (**for developers**)

  * sphinx: `https://github.com/sphinx-doc/sphinx <https://github.com/sphinx-doc/sphinx>`_ 
  * nbsphinx: `https://github.com/spatialaudio/nbsphinx <https://github.com/spatialaudio/nbsphinx>`_ 
  * sphinx-gallery: `https://github.com/sphinx-gallery/sphinx-gallery <https://github.com/sphinx-gallery/sphinx-gallery>`_ 
  * sphinx-inline-tabs: `https://github.com/pradyunsg/sphinx-inline-tabs <https://github.com/pradyunsg/sphinx-inline-tabs>`_ 
  * sphinx-intl: `https://github.com/sphinx-doc/sphinx-intl <https://github.com/sphinx-doc/sphinx-intl>`_ 
  * sphinx-rtc-theme: `https://github.com/readthedocs/sphinx_rtd_theme <https://github.com/readthedocs/sphinx_rtd_theme>`_ 
  * furo: `https://github.com/pradyunsg/furo <https://github.com/pradyunsg/furo>`_ 

* package testing and releasing (**for developers**)

  * pytest: `https://github.com/pytest-dev/pytest <https://github.com/pytest-dev/pytest>`_ 
  * packaging: `https://github.com/pypa/packaging <https://github.com/pypa/packaging>`_ 
  * wheel: `https://github.com/pypa/wheel <https://github.com/pypa/wheel>`_ 

Funding
-------

This project was partially funded by:

* the JST AIP Acceleration Research “Studies of CPS platform to raise big-data-driven AI agriculture”; 
* the SICORP Program JPMJSC16H2; 
* CREST Programs JPMJCR16O2 and JPMJCR16O1; 
* the International Science & Technology Innovation Program of Chinese Academy of Agricultural Sciences (CAASTIP); 
* the National Natural Science Foundation of China U19A2