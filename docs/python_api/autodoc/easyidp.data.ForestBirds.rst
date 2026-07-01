easyidp.data.ForestBirds
========================

.. currentmodule:: easyidp.data

.. autoclass:: ForestBirds
   :no-members:

   .. automethod:: __init__

``ForestBirds`` inherits the common dataset API from
:class:`easyidp.data.dataset.Dataset`. The inherited attributes and methods are
documented on the hidden Data Advanced API page.

**Inherited attributes**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - :attr:`name <easyidp.data.dataset.Dataset.name>`
     - Dataset manifest name.
   * - :attr:`root <easyidp.data.dataset.Dataset.root>`
     - Extracted dataset directory.

**Inherited methods**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - :meth:`download() <easyidp.data.dataset.Dataset.download>`
     - Download and extract the dataset.
   * - :meth:`dry_run() <easyidp.data.dataset.Dataset.dry_run>`
     - Return a JSON-friendly summary without touching the network.
   * - :meth:`is_ready() <easyidp.data.dataset.Dataset.is_ready>`
     - Check whether required dataset files are available.
   * - :meth:`path() <easyidp.data.dataset.Dataset.path>`
     - Resolve a dotted manifest key to an absolute path.