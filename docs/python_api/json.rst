:hide-toc:

===============
• jsonfile
===============

.. currentmodule:: easyidp.jsonfile

This is a package for simple json file operations. 
Mainly about reading and writting json files from python dict object.

It can be viewed as a submodule for the :class:`easyidp.ROI <easyidp.roi.ROI>` class.

Functions
=========

.. caution::
    
    The ``easyidp.ROI`` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need to.

.. autosummary::
    :toctree: autodoc

    read_json
    dict2json
    write_json
    save_json