:orphan:

easyidp.pointcloud.compat
=========================

.. automodule:: easyidp.pointcloud.compat
   :no-members:

Legacy Standalone Functions
---------------------------

These helpers are kept for compatibility with pre-v2.1 point-cloud IO
workflows. They emit ``FutureWarning`` and delegate to the current IO
backends. Prefer :func:`easyidp.pointcloud.read_point_cloud` and
:func:`easyidp.pointcloud.write_point_cloud` in new code.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Replacement
   * - :func:`read_las`
     - Use :func:`easyidp.pointcloud.read_point_cloud`.
   * - :func:`read_laz`
     - Use :func:`easyidp.pointcloud.read_point_cloud`.
   * - :func:`read_ply`
     - Use :func:`easyidp.pointcloud.read_point_cloud`.
   * - :func:`write_las`
     - Use :func:`easyidp.pointcloud.write_point_cloud`.
   * - :func:`write_laz`
     - Use :func:`easyidp.pointcloud.write_point_cloud`.
   * - :func:`write_ply`
     - Use :func:`easyidp.pointcloud.write_point_cloud`.

.. autofunction:: read_las

.. autofunction:: read_laz

.. autofunction:: read_ply

.. autofunction:: write_las

.. autofunction:: write_laz

.. autofunction:: write_ply

Compatibility Mixin
-------------------

``PointCloudCompatMixin`` carries deprecated instance methods inherited by
:class:`easyidp.pointcloud.PointCloud`. They are documented here as the
implementation layer; user-facing method docs live on the ``PointCloud``
class page.

.. autoclass:: PointCloudCompatMixin
   :members: crop_polygon, crop_rois, crop_point_cloud, read_point_cloud, write_point_cloud
