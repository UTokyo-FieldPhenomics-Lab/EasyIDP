===========
• Pix4D
===========

.. currentmodule:: easyidp.pix4d

Class
=====

A summary of class ``easyidp.pix4d.Pix4D``, can be simple accessed by ``easyidp.Pix4D``

.. autosummary::
    :toctree: autodoc

    Pix4D
    

Functions
=========

This module (``easyidp.pix4d``) also contains the following standard-alone functions for preocessing GeoTiff file directly.

.. caution::
    
    The :class:`easyidp.Pix4D <easyidp.pix4d.Pix4D>` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need them.

.. autosummary::
    :toctree: autodoc

    parse_p4d_project
    parse_p4d_param_folder
    read_xyz
    read_pmat
    read_cicp
    read_ccp
    read_campos_geo
    read_cam_ssk