:orphan:

===========================
easyidp.data Advanced API
===========================

This page documents dataset internals and downloader helpers for contributors
and advanced integrations. Ordinary users should prefer ``Lotus``,
``ForestBirds``, ``TestData``, and ``list_datasets`` on the main Data page.

Downloader Extension Points
===========================

.. autofunction:: easyidp.data.downloader.download_dataset

.. autofunction:: easyidp.data.downloader.safe_extract_zip

Dataset Internals
=================

.. autoclass:: easyidp.data.dataset.Dataset
   :members:

.. autoclass:: easyidp.data.dataset._PathNamespace
   :members:

.. autofunction:: easyidp.data.dataset._insert_path

.. autofunction:: easyidp.data.dataset._load_manifest

.. autofunction:: easyidp.data.dataset._validate_attr_name

.. autofunction:: easyidp.data.dataset._validate_manifest

Downloader Internals
====================

.. autofunction:: easyidp.data.downloader._download_gdrive

.. autofunction:: easyidp.data.downloader._download_modelscope

.. autofunction:: easyidp.data.downloader._fetch_modelscope_file_info

.. autofunction:: easyidp.data.downloader._file_sha256

.. autofunction:: easyidp.data.downloader._result

.. autofunction:: easyidp.data.downloader._select_mirror

.. autofunction:: easyidp.data.downloader._verify_downloaded_file
