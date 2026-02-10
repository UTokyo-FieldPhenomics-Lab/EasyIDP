# Container + ROI Refactor Core Plan (Minimal Change)

## Scope

This plan is the core reference for the upcoming refactor with two hard goals:

1. Extract `Container` from `src/easyidp/__init__.py` into a dedicated base-structures module.
2. Move shapefile key-generation logic (`name_field`, `include_title`) out of `idp.shp.read_shp` and into `ROI` APIs.

Design boundary:

- `Container`: only `int + str + slice` indexing and ordered storage.
- `ROI`: owns attributes (`attrs`) and attribute-driven key rename logic.
- `idp.shp.read_shp`: only reads and returns shapefile data.

---

## Behavior Contracts (must keep)

### Container set behavior (`__setitem__`)

- `key` is `int`:
  - `0 <= key < len(self)`: replace existing item.
  - `key == len(self)`: append item.
  - `key == len(self) + 1`: append item (compatibility mode; add warning).
  - `key == -1`: append item (user intent: assign-to-tail).
  - otherwise: raise `IndexError`.
- `key` is `str`:
  - exists: replace.
  - not exists: append new key-item pair.
- other key types: raise `TypeError`.

### Container get behavior (`__getitem__`)

- `int`: strict bounds check, raise `IndexError` when out of range.
- `str`: strict key lookup, raise `KeyError` when not found.
- `slice`: return same-class container with sliced order preserved.

### ROI attrs + rename behavior

- `_attrs` is `list[dict]`, aligned by index with container items.
- Any key-rename API must be transactional:
  - pre-compute all target keys,
  - validate conflicts,
  - apply changes in one commit step,
  - on failure, state unchanged.

---

## File-by-File Minimal Patch Plan

### 1) `src/easyidp/structures/__init__.py` (new)

Purpose: central export for custom base structures.

Pseudo diff:

```diff
+ from .container import Container, Entry
+
+ __all__ = ["Container", "Entry"]
```

### 2) `src/easyidp/structures/container.py` (new)

Purpose: hold the new lightweight ordered container.

Pseudo diff:

```diff
+ from dataclasses import dataclass
+ from copy import deepcopy
+ from typing import Any, Iterator
+ import warnings
+
+ @dataclass
+ class Entry:
+     key: str
+     item: Any
+
+ class Container:
+     + _entries: list[Entry]
+     + _key_to_idx: dict[str, int]
+
+     + __getitem__(int|str|slice)
+     + __setitem__(int|str)
+     + __delitem__(int|str)
+     + keys(), values(), items(), __contains__(), __iter__(), __len__(), copy()
+
+     + int set policy: -1 / len / len+1 => append
+     + strict get policy for missing str => KeyError
```

### 3) `src/easyidp/__init__.py` (modify)

Purpose: remove embedded container implementation and re-export new module.

Pseudo diff:

```diff
- class Container(dict):
-     ...
+ from .structures import Container
```

Notes:

- Keep `idp.Container` import path stable to minimize external breakage.
- Remove no-longer-needed helper `_find_key` only if not used elsewhere.

### 4) `src/easyidp/shp.py` (modify)

Purpose: simplify `read_shp` to reader-only logic.

Pseudo diff:

```diff
- def read_shp(..., name_field=-1, include_title=False, ...):
+ def read_shp(..., encoding="utf-8", return_proj=False):
     ...
-    # name_field/include_title template + key generation
-    # duplicate generated key checks
-    shp_dict[plot_name] = coord_np
+    polygons.append(coord_np)
+    records.append(record_as_dict)
+    fields = _get_field_key(...)
     ...
-    return shp_dict[, shp_proj]
+    return polygons, records, fields[, shp_proj]
```

Notes:

- Keep helper functions for field-resolution available for ROI reuse if useful.
- If temporary compatibility needed, add deprecation warning for old args.

### 5) `src/easyidp/roi.py` (modify, core migration)

Purpose: own attrs and naming logic.

Pseudo diff:

```diff
  class ROI(idp.Container):
+     self._attrs: list[dict]

  def read_shp(..., name_field=-1, include_title=False, ...):
-     roi_dict, crs = idp.shp.read_shp(...)
-     for k, v in roi_dict.items():
-         self[k] = v
+     polygons, records, fields, crs = idp.shp.read_shp(..., return_proj=True)
+     self.clear()
+     self._attrs = []
+     for i, poly in enumerate(polygons):
+         self[str(i)] = poly
+         self._attrs.append(records[i])
+     self.rename_by_fields(name_field, include_title, fields)

+ def rename_by_fields(name_field=-1, include_title=False, fields=None): ...
+ def _build_key_from_attrs(attrs, idx, name_field, include_title, fields): ...
+ def get_attrs(key_or_idx): ...
+ def set_attrs(key_or_idx, attrs): ...
+ def update_attrs(key_or_idx, patch): ...
+ def rename_by_attrs(selector, rename_fn): ...
```

Notes:

- Naming rule parity must follow old `name_field/include_title` behavior.
- `name_field` accepted forms: `str | int | list[str|int] | "#"`.
- Sanitize generated key: replace `/` and `\\` with `_`.
- Duplicate generated keys: `KeyError`.

### 6) `src/easyidp/reconstruct.py` (modify)

Purpose: remove redundant old-field initialization.

Pseudo diff:

```diff
  class ProjectPool(idp.Container):
      def __init__(self):
          super().__init__()
-         self.id_item = {}
-         self.item_label = {}
```

### 7) `src/easyidp/metashape.py` (modify)

Purpose: remove direct dependency on old `item_label` internals.

Pseudo diff:

```diff
- if sensor.label in sensors.item_label.keys():
+ if sensor.label in sensors:
```

### 8) `src/easyidp/geotiff.py` (modify)

Purpose: avoid direct copy of removed internals.

Pseudo diff:

```diff
- roi_obj.id_item = roi.id_item.copy()
- roi_obj.item_label = roi.item_label.copy()
+ roi_obj = roi.copy()
```

---

## Test Migration Plan (API-aware)

### 1) `tests/test_init_class_func.py`

- Replace assertions that directly check `id_item/item_label`.
- Assert public behavior instead:
  - `keys()`, `values()`, `items()` order and content,
  - `container[int]`, `container[str]`, `slice`, delete reindex.
- Add set-edge tests:
  - `container[-1] = item` appends,
  - `container[len(container)+1] = item` appends,
  - invalid larger index raises `IndexError`.

### 2) `tests/test_shp.py`

- Move/remove tests that verify key rename from `read_shp` output.
- Keep only reader-focused tests:
  - geometry extraction,
  - records extraction,
  - fields mapping,
  - projection return behavior.

### 3) `tests/test_roi.py`

- Add/move rename behavior tests here (from `test_shp.py`):
  - `name_field` with `str/int/list/#`,
  - `include_title` true/false,
  - duplicate key detection.
- Add attrs alignment tests:
  - after load,
  - after delete and reindex,
  - after slice/copy.
- Add transactional test for `rename_by_attrs` rollback on conflict.

---

## Error Handling Matrix

### Container

- `TypeError`: unsupported key types for get/set/delete.
- `IndexError`: int get/set out of bounds except append-compatible cases.
- `KeyError`: missing string key on get/delete.
- `KeyError`: duplicate key during rename.

### ROI naming APIs

- `KeyError`: `name_field` string missing in fields.
- `IndexError`: `name_field` int out of range.
- `KeyError`: generated key collision.
- `ValueError`: internal state mismatch (`len(_attrs) != len(container)`).

---

## Execution TODO List (for implementation phase)

- [ ] Create `src/easyidp/structures/` package and new `container.py` implementation.
- [ ] Replace `Container` in `src/easyidp/__init__.py` with re-export import.
- [ ] Refactor `src/easyidp/shp.py::read_shp` to reader-only output.
- [ ] Implement ROI attrs-by-idx and field-based rename APIs in `src/easyidp/roi.py`.
- [ ] Update `src/easyidp/metashape.py` and `src/easyidp/geotiff.py` old-internal references.
- [ ] Clean up `src/easyidp/reconstruct.py` redundant old state init.
- [ ] Migrate tests across `test_init_class_func.py`, `test_shp.py`, `test_roi.py`.
- [ ] Run targeted tests first, then full suite:
  - `uv run pytest tests/test_init_class_func.py -q`
  - `uv run pytest tests/test_shp.py -q`
  - `uv run pytest tests/test_roi.py -q`
  - `uv run pytest`

---

## Commit Strategy (minimal risk)

1. `refactor(structures): extract Container to dedicated module`
2. `refactor(roi,shp): move key-generation from read_shp to ROI`
3. `test: migrate shp/roi/container behavior tests`

This ordering keeps each change-set reviewable and easier to rollback.
