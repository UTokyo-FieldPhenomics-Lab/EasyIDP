# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../easyidp"))

import easyidp

# -- Project information -----------------------------------------------------

project = 'EasyIDP'
copyright = '2022, Haozhou Wang'
author = 'Haozhou Wang'

# The full version, including alpha/beta/rc tags
release = easyidp.__version__

print(easyidp.__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # support numpy docs string
    'sphinx.ext.todo',
    "sphinx_inline_tabs",
    "sphinx.ext.autosummary",
    "nbsphinx",  # jupyte notebook
    'sphinx_gallery.load_style',  # gallery view in notebooks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# -- bilingual translation ----
locale_dirs = ['locale/']  
# language for local preview
language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = "EasyIDP 2.0"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

todo_include_todos = True

#---------------
# numpy documentations
napoleon_google_docstring = False
napoleon_use_admonition_for_references = True

# deprecated
# autodoc_default_flags = ['members']
autodoc_default_options = {'members': True,}
autosummary_generate = True

nbsphinx_thumbnails = {
    'jupyter/load_roi': r'_static/images/jupyter/shp_icon.png',
    'jupyter/crop_outputs': r"_static/images/jupyter/crop.png",
    'jupyter/backward_projection': r"_static/images/jupyter/backward.png",
    'jupyter/get_z_from_dsm': r"_static/images/jupyter/zval.png",
}