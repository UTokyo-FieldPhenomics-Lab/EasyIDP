# EasyIDP Documentation API Layout Rules

Use this file as the source of truth for API documentation layout. The goal is
reader-oriented API pages that keep common workflows visible while moving
advanced, inherited, compatibility, and contributor-facing details out of the
main reading path.

## Module Pages

- Prefer reader-oriented single-page API docs over one-page-per-function fragmentation.
- Each user-facing module page should keep ordinary public `Classes` and `Functions` sections.
- Small public functions should be expanded inline with `.. autofunction::` on the module page. Do not keep one standalone `autodoc/easyidp.module.function.rst` page for ordinary helpers.
- Use `autosummary :toctree: autodoc` for a class itself when the class deserves its own page, but avoid `:toctree:` for ordinary methods and functions unless a function is large enough to justify a standalone tutorial-like reference page.
- Module pages may add a bottom `Advanced API` section for hidden, implicit, private, compatibility, or contributor-facing classes/functions. Maintain Advanced API from the module page only, not from each class or submodule page.

## Class And Submodule Pages

- Core public classes should have one class page where methods and attributes are summarized and expanded inline.
- Class pages should list the class API only; hidden, private, implicit, compatibility, or contributor-facing APIs are handled from the parent module page or a linked Advanced API page.
- Class and submodule pages should list all APIs that belong to that page. When a page has more than about six functions, methods, or attributes, add top-of-page categorized jump tables such as `Attributes`, `Methods`, and, when applicable, compatibility groups.
- Jump-table rows should link to same-page anchors when the API is documented on that page. Inherited or shared APIs may link to their canonical base-class or Advanced API anchors instead.
- Keep class-page order predictable: class name, categorized API tables, `__init__`, then functions and attributes in alphabetical order unless a reader-facing reason justifies another order.
- When a public subclass mostly specializes construction or dataset-specific paths, do not duplicate inherited base-class APIs on every subclass page. Add a concise inherited-API table that links to the canonical base-class anchors instead.
- Avoid RST section headings for secondary explanatory blocks inside generated class pages when they would pollute the left sidebar. Prefer `.. rubric::`, bold text, plain paragraphs, or tables for non-navigation content.

## Advanced API Pages

- `Advanced API` should prefer module-level single pages such as `compat.html`, `geometry.html`, `io.html`, or `easyidp.data.advanced.html`; each page should summarize and expand all related helper functions/classes inline.
- Advanced API pages may live under `docs/python_api/autodoc/` with names such as `easyidp.data.advanced.rst`; mark them with `:orphan:` and link them only from the owning module page.
- Do not add advanced autodoc pages to the root `docs/index.rst` toctree; they should be reachable from the module page but not shown in the left sidebar.
- Do not generate one standalone page per advanced helper function or private method. Keep those entries inline inside their advanced module page.
- In `Advanced API`, list explicit extension points before implicit/private helpers so readers see stable integration points first.
- Advanced API docstrings still need the same quality as main APIs: clear purpose, parameters, returns, notes when useful, and examples when the object is user- or contributor-facing.

## Sphinx Templates And Links

- Use custom `autosummary` templates when they reduce manual maintenance for class or submodule pages, for example to auto-list public methods and attributes from Sphinx template variables.
- Template content that belongs to a class must be indented inside the `.. autoclass::` directive so generated HTML keeps methods and attributes inside the class block.
- Custom autosummary templates should live in `docs/_templates/autosummary/` and be referenced as `:template: autosummary/<template>.rst`.
- API tables may use short visible labels with Sphinx explicit-title links, for example `:meth:`download() <easyidp.data.dataset.Dataset.download>`, so readers see concise names while links still target stable anchors.
