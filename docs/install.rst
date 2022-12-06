.. install:

===============
Install EasyIDP
===============


EasyIDP Python packages are distributed via
`PyPI <https://pypi.org/project/easyidp/>`_.


Supported Python versions:

* 3.8
* 3.9

Supported operating systems:

* Windows 10 (64-bit)
* Windows 11 (64-bit)

The following operating system should also works, but have not been tested:

* Ubuntu 18.04+
* macOS 10.15+

Using this package requires basic Python programming knowledge, including basic python gramma and package installation.


Install from PyPI
=================

Here assuming the user has python3 and pip installed in the computer, and in the command line, the following operation works:

.. code-block:: bash

    C:\Users\xxx> python --version
    Python 3.7.3
    C:\Users\xxx> pip --version
    pip 19.2.3

Then install the package by:

.. code-block:: bash

    pip install easyidp


.. note::
    In general, we recommend using a
    `virtual environment <https://docs.python-guide.org/dev/virtualenvs/>`_
    or `conda environment <https://docs.conda.io/en/latest/miniconda.html>`_ to avoid dependices conflict with your local environment.

    Please refer :doc:`backgrounds/virtualenv` for more details.

    For Linux users, depending on the configurations, ``pip3`` may be needed for
    Python 3, or the ``--user`` option may need to be used to avoid permission
    issues. For example:

    .. code-block:: bash

        pip3 install easyidp
        # or
        pip install --user easyidp
        # or
        python3 -m pip install --user easyidp


.. tip::
    For users in China, it is recommended to use Tsinghua source to accelerate the download speed:

    .. code-block:: bash

        > pip install easyidp -i https://pypi.tuna.tsinghua.edu.cn/simple


Using from source code
======================

If you need to make some changes to the source code (e.g. fix bugs) and want it work immediately (rather than waiting for offical fix). You can also using from the source code directly.

.. tip::
    Please ensure you have uninstalled the pypi easyidp in your current environment:

    .. code-block:: bash

        pip uninstall easyidp

    and need to restart python to make changes taking effects.


Assuming the source package was downloaded in ``C:\path\to\source\code`` and the ``code`` folder has the following files:

.. code-block:: text

    C:\path\to\source\code
    ├─ docs/
    ├─ easyidp/
    ├─ tests/
    readme.md
    setup.py
    ...

Then you can used the following code to manual import easyidp package:

.. code-block:: python

    import sys
    sys.path.insert(0, r'C:/path/to/source/code/EasyIDP')

    import easyidp as idp

Or install to your virtual environment by:

.. code-block:: bash

    > cd "C:/path/to/source/code/EasyIDP"   # contains setup.py
    > pip install -e .