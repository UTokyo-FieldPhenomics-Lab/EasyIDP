============
reconstruct
============

.. currentmodule:: easyidp.reconstruct

.. caution:: 

    This is the base module for 3D reconstruction project, all the following classes and functions have already been implemented into the submodule :class:`easyidp.Pix4D <easyidp.pix4d.Pix4D>` and :class:`easyidp.Metashape <easyidp.metashape.Metashape>`. They are generally sufficient for most application cases, it is not recommended to create the following classes or use the following functions directly unless you really need to.


Class
=====

A summary of base class in the module ``easyidp.reconstruct``. 

.. autosummary::
    :toctree: autodoc

    Recons
    Sensor
    Photo
    Calibration
    ChunkTransform

You can definately access these base class directly by: 

.. code-block:: python

    >>> import easyidp as idp

    >>> sensor = idp.reconstruct.Sensor()
    >>> sensor
    <easyidp.reconstruct.Sensor object at 0x7fdc000450d0>

    >>> photo = idp.reconstruct.Photo()
    >>> photo
    <easyidp.reconstruct.Photo object at 0x7fdc40996190>

But it is more often used in the Pix4D or Metashape project in this way:

.. tab:: Pix4D

    Load the example data:

    .. code-block:: python

        >>> test_data = idp.data.TestData()

    And read the demo pix4d project:

    .. code-block:: python
        
        >>> p4d = idp.Pix4D(
        ...     project_path=test_data.pix4d.lotus_folder,
        ...     param_folder=test_data.pix4d.lotus_param
        ... )

    Then access the classes:

    .. code-block:: python

        >>> p4d.sensors
        <easyidp.Container> with 1 items
        [0]     FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456
        <easyidp.reconstruct.Sensor object at 0x7fdc40996580>

        >>> p4d.photos[0]
        <easyidp.reconstruct.Photo object at 0x7fdc40996220>

.. tab:: Metashape

    Load the example data:

    .. code-block:: python

        >>> test_data = idp.data.TestData()

    And read the demo metashape project:

    .. code-block:: python

        >>> ms = idp.Metashape(test_data.metashape.lotus_psx, chunk_id=0)

    Then access the classes:

    .. code-block:: python

        >>> ms.photos
        <easyidp.Container> with 151 items
        [0]     DJI_0422
        <easyidp.reconstruct.Photo object at 0x7fdc409a4040>
        [1]     DJI_0423
        <easyidp.reconstruct.Photo object at 0x7fdc40996c70>
        ...
        [149]   DJI_0571
        <easyidp.reconstruct.Photo object at 0x7fdc103e2910>
        [150]   DJI_0572
        <easyidp.reconstruct.Photo object at 0x7fdc103e2940>



Functions
=========

.. autosummary::
    :toctree: autodoc

    sort_img_by_distance
    save_back2raw_json_and_png