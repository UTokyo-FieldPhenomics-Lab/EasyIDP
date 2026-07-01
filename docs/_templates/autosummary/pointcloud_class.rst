{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% set compatibility_methods = ["crop_point_cloud", "crop_polygon", "crop_rois", "read_point_cloud", "write_point_cloud"] %}

.. autoclass:: {{ objname }}
   :no-members:

{% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
{% for item in attributes | sort %}
      ~{{ objname }}.{{ item }}
{% endfor %}
{% endif %}

{% if methods %}
   .. rubric:: Methods

   .. autosummary::
{% for item in methods | sort %}
{% if item != "__init__" and item not in compatibility_methods %}
      ~{{ objname }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}

   .. automethod:: __init__

{% for item in (attributes + methods) | sort %}
{% if item != "__init__" and item not in compatibility_methods %}
{% if item in attributes %}
   .. autoattribute:: {{ item }}
{% else %}
   .. automethod:: {{ item }}
{% endif %}

{% endif %}
{% endfor %}
