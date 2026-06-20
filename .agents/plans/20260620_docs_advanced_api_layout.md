# Docs Advanced API Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize hidden/internal API documentation under each module page and add missing config API docs.

**Architecture:** Keep normal public `Classes` and `Functions` autosummary sections near the main module content. Add a bottom `Advanced API` section for hidden or contributor-facing classes/functions, split into `Classes` and `Functions` tables when both kinds exist. Keep generated advanced pages under `docs/python_api/autodoc/`, mark them `:orphan:`, and do not add them to any toctree so Furo does not show them in the left sidebar.

**Tech Stack:** Sphinx, `sphinx.ext.autosummary`, `sphinx.ext.autodoc`, reStructuredText.

---

### Task 1: Document The Advanced API Rule

**Files:**
- Modify: `.agents/rules/architecture.md`

- [ ] **Step 1: Add a documentation policy subsection**

Add a concise docs rule stating:

```markdown
## Documentation API Layout Policy

- Each module page should keep ordinary public `Classes` and `Functions` autosummary sections for user-facing APIs.
- Public `Classes` and `Functions` sections may use `autosummary :toctree: autodoc` so user-facing API pages appear in the module's left sidebar.
- Module pages may add a bottom `Advanced API` section for hidden, implicit, private, or contributor-facing classes/functions.
- `Advanced API` should be split into `Classes` and `Functions` autosummary tables when both kinds exist.
- In `Advanced API`, list explicit classes/functions before implicit/private helpers so readers see stable extension points first.
- `Advanced API` autosummary tables must not use `:toctree:`; otherwise Furo will show hidden/private pages in the left sidebar.
- Create explicit autodoc stub pages for `Advanced API` entries and mark those pages with `:orphan:` so they are linkable but not shown in the sidebar toctree.
- Advanced API entries should generate clickable autodoc pages with the same docstring quality expectations as main APIs: clear purpose, parameters, returns, notes when useful, and examples when the object is user- or contributor-facing.
- Do not add advanced autodoc pages to the root `docs/index.rst` toctree; they should be reachable from the module page but not shown in the left sidebar.
```

- [ ] **Step 2: Search for duplicate/conflicting docs rules**

Run: `rg "Advanced API|Documentation API Layout|autosummary" .agents docs -g '*.md' -g '*.rst'`

Expected: no conflicting policy text.

### Task 2: Add Config API Page

**Files:**
- Modify: `docs/index.rst`
- Modify: `docs/python_api/index.rst`
- Create: `docs/python_api/config.rst`
- Create: `docs/python_api/autodoc/easyidp.config.EasyIDPConfig.rst`
- Create: `docs/python_api/autodoc/easyidp.config.get.rst`
- Create: `docs/python_api/autodoc/easyidp.config.set.rst`
- Create: `docs/python_api/autodoc/easyidp.config.reset.rst`
- Create: `docs/python_api/autodoc/easyidp.config.default_config_path.rst`
- Create: `docs/python_api/autodoc/easyidp.config.default_data_dir.rst`

- [ ] **Step 1: Put config before data in navigation**

In `docs/index.rst`, replace the Python API toctree sequence:

```rst
   python_api/index
   python_api/data
   python_api/advanced
```

with:

```rst
   python_api/index
   python_api/config
   python_api/data
```

- [ ] **Step 2: Update API summary list**

In `docs/python_api/index.rst`, replace the old Advanced Notes entry with:

```rst
- :doc:`Config Module <./config>` : package-wide settings for data directory, logging level, and startup banner.
- :doc:`Data Module <./data>` : Optional official demo-data path shortcuts for examples and tutorials.
```

- [ ] **Step 3: Create config module page**

Create `docs/python_api/config.rst` with public class/function sections and a bottom Advanced API section:

```rst
======
Config
======

.. currentmodule:: easyidp.config

Purpose
=======

The config module stores package-wide EasyIDP preferences in a small JSON file.
It is the single public entry point for settings such as the demo dataset root,
logger level, and startup banner display.

Most users should access it through ``easyidp.config``:

.. code-block:: python

    import easyidp as idp

    idp.config.set(data_dir="/path/to/easyidp.data")
    data_dir = idp.config.get("data_dir")

Settings
========

``data_dir``
    Root folder for EasyIDP demo datasets.

``log_level``
    Logger level used by EasyIDP, such as ``"INFO"`` or ``"DEBUG"``.

``show_banner``
    Whether EasyIDP shows the startup banner during import.

Classes
=======

.. autosummary::
    :toctree: autodoc

    EasyIDPConfig

Functions
=========

.. autosummary::
    :toctree: autodoc

    get
    set
    reset

Advanced API
============

These helpers are mainly useful for contributors and advanced users who need
to inspect EasyIDP's platform-specific default locations.

Functions
---------

.. autosummary::

    default_config_path
    default_data_dir
```

- [ ] **Step 4: Add autodoc stub pages for config APIs**

Use `.. autofunction::` for functions and `.. autoclass::` with members for `EasyIDPConfig`. Mark only the Advanced API helper pages with `:orphan:`.

### Task 3: Move Data Advanced Notes Into Data Page

**Files:**
- Modify: `docs/python_api/data.rst`
- Delete: `docs/python_api/advanced.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset.Dataset.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset._PathNamespace.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset._load_manifest.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset._validate_attr_name.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset._validate_manifest.rst`
- Create: `docs/python_api/autodoc/easyidp.data.dataset._insert_path.rst`

- [ ] **Step 1: Append Advanced API to data page**

After the existing `Functions` section, add:

```rst
Advanced API
============

``easyidp.data`` builds short demo-data attributes from JSON manifest keys.
Dotted keys such as ``metashape.project`` and ``metashape.outputs.dom`` are
expanded into runtime namespaces so users can write ``lotus.metashape.project``
or ``lotus.metashape.outputs.dom``.

The recursive namespace object is implemented as
``easyidp.data.dataset._PathNamespace``. The objects below are intended for
advanced users and contributors who need to understand manifest parsing,
runtime path expansion, and dataset validation. They are not exported from
``easyidp.data`` unless shown in the public sections above.

.. autosummary::

    dataset.Dataset
    dataset._PathNamespace

Functions
---------

.. autosummary::

    dataset._load_manifest
    dataset._validate_attr_name
    dataset._validate_manifest
    dataset._insert_path
```

- [ ] **Step 2: Delete old advanced page**

Remove `docs/python_api/advanced.rst`.

- [ ] **Step 3: Add advanced data autodoc stubs**

Use fully qualified `.. autoclass:: easyidp.data.dataset.Dataset` and `.. autofunction:: easyidp.data.dataset._load_manifest` patterns. Add `:orphan:` to every data Advanced API stub page.

### Task 4: Verify References And Buildability

**Files:**
- Verify only.

- [ ] **Step 1: Search stale advanced references**

Run: `rg "python_api/advanced|Advanced Notes|./advanced" docs .agents`

Expected: no matches.

- [ ] **Step 2: Search data/config API references**

Run: `rg "config\.rst|Config Module|Advanced API|_PathNamespace|easyidp\.config" docs/python_api .agents/rules/architecture.md`

Expected: config page and data advanced section are present.

- [ ] **Step 3: Run focused tests**

Run: `uv run pytest tests/test_config tests/test_data`

Expected: all pass.

- [ ] **Step 4: Build docs if dependencies are available**

Run: `uv run sphinx-build -b html docs docs/_build/html`

Expected: build succeeds. If it fails due to missing optional docs dependencies or existing warnings, report exact failure.
