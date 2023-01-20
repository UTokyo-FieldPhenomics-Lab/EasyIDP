============
Data
============

.. currentmodule:: easyidp.data

Dataset
=======

The datasets are as follows (**for user**):

.. autosummary::
    :toctree: autodoc

    Lotus

Use example:

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

.. caution::

    For Chinese mainland user who can not access the GoogleDrive directly, please manually download the dataset by the following way:

    .. code-block:: python

        >>> cn_url = idp.data.Lotus.url_list[1]
        >>> cn_rul
        'https://fieldphenomics.cowtransfer.com/s/9a87698f8d3242'

    Please download the zip file in previous link, and unzip them into the following folder:

    .. code-block:: python

        >>> idp.data.show_data_dir()

    It will call your local file explorer to show the folder to put the data. Please insure the folder name the same with the dataset name:

    .. code-block:: python

        >>> idp.data.Lotus.name
        '2017_tanashi_lotus'

    And the folder should have the following structure:

    .. tab:: Windows

        .. code-block:: text

            . C:\Users\<user>\AppData\Local\easyidp.data
            |-- 2017_tanashi_lotus
            |-- gdown_test
            |-- ...

    .. tab:: MacOS

        .. code-block:: text

            . ~/Library/Application Support/easyidp.data
            |-- 2017_tanashi_lotus
            |-- gdown_test
            |-- ...

    .. tab:: Linux/BSD

        .. code-block:: text

            . ~/.local/share/easyidp.data   # or in $XDG_DATA_HOME, if defined
            |-- 2017_tanashi_lotus
            |-- gdown_test
            |-- ...

    Or download all dataset from `this link <https://fieldphenomics.cowtransfer.com/s/25f92eb0585b4d>`_ at once, and unzip to previous folder structure.


The dataset base class and testing class (**for developers**): 

.. autosummary::
    :toctree: autodoc

    EasyidpDataSet
    TestData


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