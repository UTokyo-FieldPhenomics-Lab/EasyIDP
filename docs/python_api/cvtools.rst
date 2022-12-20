===========
• cvtools
===========

.. currentmodule:: easyidp.cvtools

This is a package for simple imarray operations. 
Mainly about creating binary masks and cropping image parts from given polygon coordinates.

It can be viewed as a submodule for the :class:`easyidp.GeoTiff <easyidp.geotiff.GeoTiff>` class.

Functions
=========

.. autosummary::
    :toctree: autodoc

    imarray_crop
    poly2mask
    rgb2gray