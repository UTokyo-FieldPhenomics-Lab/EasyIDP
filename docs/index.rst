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

.. image:: https://img.shields.io/tokei/lines/github/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic
   :alt: GitHub code size in bytes

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



Coordinates Order
=================

In the EasyIDP, we use the (horizontal, vertical, dim) order as the coords order. When it applies to the GIS coordintes, is (longitude, latitude, altitude)


Examples
========

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   install
   jupyter/quick_start

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

   jupyter/quick_start

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Python API

   python_api/index
   python_api/pointcloud
   python_api/geotiff
   python_api/cvtools
   python_api/roi
   python_api/shp
   python_api/json
   python_api/data


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contribute

.. Getting Started
.. ==================

.. * :doc:`install`
.. * :doc:`contribute`