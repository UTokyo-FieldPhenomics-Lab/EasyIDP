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
   :target: https://img.shields.io/

.. image:: https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic
   :alt: GitHub
   :target: https://img.shields.io/

.. image:: https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic
   :alt: GitHub top language

.. admonition:: other languages
   
   This is a multi-language document, you can change document languages here or at the bottom left corner.

   .. hlist::
      :columns: 3

      * `English <https://easyidp.readthedocs.io/en/latest/>`_
      * `中文 <https://easyidp.readthedocs.io/zh_CN/latest/>`_
      * `日本語(翻訳募集) <https://easyidp.readthedocs.io/ja/latest/>`_


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


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Python API

   python_api/index
   python_api/pointcloud


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contribute

.. Getting Started
.. ==================

.. * :doc:`install`
.. * :doc:`contribute`