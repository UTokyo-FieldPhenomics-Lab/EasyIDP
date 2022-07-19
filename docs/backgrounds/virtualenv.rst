.. virtualenv:

=======================
Use Virtual Environment
=======================

Please choose either ``virtualenv`` python package or ``conda``.

virtualenv
==========

Install virtual env manager
---------------------------

.. code-block:: bash

    C:\Users\xxx> pip install virtualenv


Create virtual env
------------------

.. code-block:: bash

    C:\Users\xxx> cd C:\path\to\env\folder
    C:\path\to\env\folder> virtualenv easyidp

Activate virtual env
--------------------

.. tab:: Windows

    .. code-block:: bash

        C:\path\to\env\folder> easyidp\Scripts\activate.bat
        (easyidp) C:\path\to\env\folder>


.. tab:: Linux/BSD

    .. code-block:: bash

        path/to/env/folder$ source easyidp/bin/activate
        (easyidp) path/to/env/folder$

Install EasyIDP package
------------------------

.. code-block:: bash

    (easyidp)> pip install easyidp


Exit the virtual env
--------------------

.. code-block:: bash

    # exit the env 
    (easyidp)> deactivate


Delete the virtual env
----------------------

Delete the folder ``C:\path\to\env\folder`` directly to delete environment


---------------------


Conda
=====

Create conda env
----------------

.. code-block:: bash

    C:\User\xxx> conda create -n easyidp python=3.8

Activate conda env
------------------

.. code-block:: bash

    C:\User\xxx> conda activate EasyIDP
    (easyidp) C:\User\xxx>


Install EasyIDP package
-----------------------

Due to the EasyIDP package have not been published to Conda forge, you could only use pip to install.

.. code-block:: bash

    (easyidp)C:\User\xxx> pip install easyidp

.. tip::
    If you using pip in conda virtual envs, other packages should also installed by pip. Do not use ``conda install`` and ``pip install`` at he same time.

Exit conda env
--------------

.. code-block:: bash

    (EasyIDP) C:\User\xxx> conda deactivate
    C:\User\xxx>

Delete conda env
----------------

.. code-block:: bash

    C:\User\xxx> Conda remove -n easyidp --all