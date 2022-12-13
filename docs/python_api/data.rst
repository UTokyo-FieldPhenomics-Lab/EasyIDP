============
Data
============

.. currentmodule:: easyidp.data

Dataset
=======


Here have the following dataset: 

.. autosummary::
    :toctree: autodoc

    EasyidpDataSet
    TestData
    Lotus

Dataset use examples:

.. code-block:: python

    >>> import easyidp as idp
    >>> lotus = idp.data.Lotus()
    Downloading...
    From: https://drive.google.com/uc?id=1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j
    To: C:\Users\<user>\AppData\Local\easyidp.data\2017_tanashi_lotus.zip
    100%|█████████████████████████████| 3.58G/3.58G [00:54<00:00, 65.4MB/s]
    >>> lotus.shp
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\plots.shp'
    >>> lotus.metashape.proj
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\170531.Lotus.psx'
    >>> lotus.photo
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\photos'
    >>> lotus.pix4d.param
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\params'


Functions
=========

.. autosummary:: 
    :toctree: autodoc

    user_data_dir
    show_data_dir
    url_checker
    download_all


The functions can be used by:

.. code-block:: python

    >>> import easyidp as idp
    >>> idp.data.user_data_dir()
    PosixPath('/Users/<user>/Library/Application Support/easyidp.data')
    >>> idp.data.show_data_dir()
    # please check the popup file explorer