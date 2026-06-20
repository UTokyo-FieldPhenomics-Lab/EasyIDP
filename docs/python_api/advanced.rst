========
Advanced
========

Data Internals
==============

``easyidp.data`` builds short demo-data attributes from JSON manifest keys. Dotted keys such as ``metashape.project`` and ``metashape.outputs.dom`` are expanded into runtime namespaces so users can write ``lotus.metashape.project`` or ``lotus.metashape.outputs.dom``.

The recursive namespace object is implemented as ``easyidp.data.dataset._PathNamespace``. It is an internal helper for advanced users and contributors who need to understand manifest parsing. It is intentionally not exported from ``easyidp.data`` and should not be treated as a stable public API.
