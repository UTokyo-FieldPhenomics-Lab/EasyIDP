# back2raw2geotiff Memory Guard & Iterator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `back2raw2geotiff` 增加大任务告警与 iterator 返回模式，在不破坏默认行为的前提下，降低大数据场景内存风险。

**Architecture:** 将 `return_iterator` 设计为三态：`None`（自动模式，默认）、`True`（强制 iterator）、`False`（强制 dict）。新增 `return_iterator` 路径用于流式消费；在任务提交前使用简化规则：当 `return_iterator is None`、`num_workers is None` 且待处理数量超过 2000 时，输出 warning 并自动切换到 iterator。若用户显式传入 `False`，则即使超过 2000 也不自动切换。

**Tech Stack:** Python, concurrent.futures, pytest

---

### Task 1: API 设计与兼容性落地

**Files:**

- Modify: `src/easyidp/geotiff.py`
- Test: `tests/test_geotiff.py`

- [ ] **Step 1: 写 failing test（新参数默认兼容）**

```python
def test_back2raw2geotiff_default_behavior_compatible(shared_data):
    p4d = shared_data["p4d"]
    roi = shared_data["roi"]
    out_all = shared_data["out_all"]

    result = idp.geotiff.back2raw2geotiff(
        recons=p4d,
        back2raw_result=out_all,
        roi=roi,
    )

    assert isinstance(result, dict)
```

- [ ] **Step 2: 运行 test 并确认失败（或先通过用于回归基线）**

Run: `uv run pytest tests/test_geotiff.py -k "default_behavior_compatible"`

- [ ] **Step 3: 扩展函数签名（最小实现）**

```python
def back2raw2geotiff(
    recons,
    back2raw_result,
    roi,
    output_folder=None,
    nodata=0,
    has_alpha=True,
    use_affine=False,
    num_workers=None,
    return_iterator=None,
    iterator_ordered=True,
):
    ...
```

- [ ] **Step 4: 运行 test 确认兼容**

Run: `uv run pytest tests/test_geotiff.py -k "default_behavior_compatible"`

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/geotiff.py tests/test_geotiff.py
git commit -m "feat(geotiff): add iterator-compatible API parameters"
```

### Task 2: 大任务数量 warning + 自动切换（简化规则）

**Files:**

- Modify: `src/easyidp/geotiff.py`
- Test: `tests/test_geotiff.py`

- [ ] **Step 1: 写 failing test（满足简化条件时输出 warning）**

```python
def test_back2raw2geotiff_auto_switches_to_iterator_on_large_task_count(monkeypatch, tmp_path):
    warnings = []

    def fake_warning(msg, *args, **kwargs):
        warnings.append(str(msg).format(*args, **kwargs))

    monkeypatch.setattr(idp.geotiff.logger, "warning", fake_warning)
    # 构造 > 2000 个待处理任务，且 return_iterator=None + num_workers=None
    result = idp.geotiff.back2raw2geotiff(..., return_iterator=None, num_workers=None)

    assert any("数据量大且超过系统资源占用" in w for w in warnings)
    assert any("num_workers" in w and "return_iterator" in w for w in warnings)
    assert hasattr(result, "__iter__")
    assert not isinstance(result, dict)


def test_back2raw2geotiff_no_auto_switch_when_return_iterator_false(...):
    result = idp.geotiff.back2raw2geotiff(..., return_iterator=False, num_workers=None)
    assert isinstance(result, dict)
```

- [ ] **Step 2: 运行 test 并确认失败**

Run: `uv run pytest tests/test_geotiff.py -k "auto_switches_to_iterator_on_large_task_count or no_auto_switch_when_return_iterator_false"`

- [ ] **Step 3: 实现简化条件判断与 warning**

```python
def _resolve_return_iterator_mode(back2raw_result: dict, num_workers, return_iterator):
    # True/False 为用户显式指定，直接返回
    if return_iterator is True:
        return True
    if return_iterator is False:
        return False

    # return_iterator is None -> 自动模式
    if num_workers is not None:
        return False

    task_count = sum(len(v) for v in back2raw_result.values())
    if task_count <= 2000:
        return False

    default_workers = "auto"
    try:
        default_workers = multiprocessing.cpu_count()
    except Exception:
        pass

    logger.warning(
        "数据量大且超过系统资源占用。either手动减少num_workers（默认为{}）或使用return_iterator。",  # should be English
        default_workers,
    )
    return True
```

```python
# back2raw2geotiff 内调用（在 final_tasks 就绪后）
resolved_return_iterator = _resolve_return_iterator_mode(
    back2raw_result,
    num_workers,
    return_iterator,
)
```

```python
def test_back2raw2geotiff_no_warn_when_iterator_enabled(...):
    ...


def test_back2raw2geotiff_no_warn_when_num_workers_explicit(...):
    ...


def test_back2raw2geotiff_no_warn_when_task_count_small(...):
    ...
```

- [ ] **Step 4: 运行 test 确认通过**

Run: `uv run pytest tests/test_geotiff.py -k "large_task_count or auto_switch or no_auto_switch or no_warn_when"`

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/geotiff.py tests/test_geotiff.py
git commit -m "feat(geotiff): add large-task warning for iterator recommendation"
```

### Task 3: iterator 模式实现（iterator_ordered 默认 True）

**Files:**

- Modify: `src/easyidp/geotiff.py`
- Test: `tests/test_geotiff.py`

- [ ] **Step 1: 写 failing tests（返回可迭代对象 + 顺序语义）**

```python
def test_back2raw2geotiff_return_iterator_type(...):
    result = idp.geotiff.back2raw2geotiff(..., return_iterator=True)
    assert hasattr(result, "__iter__")
    assert not isinstance(result, dict)


def test_back2raw2geotiff_iterator_ordered_true(...):
    it = idp.geotiff.back2raw2geotiff(..., return_iterator=True, iterator_ordered=True)
    records = list(it)
    # 按 final_tasks 顺序断言 img_id
    assert [r[1] for r in records] == ["img1", "img2"]
```

- [ ] **Step 2: 运行 tests 并确认失败**

Run: `uv run pytest tests/test_geotiff.py -k "return_iterator_type or iterator_ordered_true"`

- [ ] **Step 3: 实现 iterator 路径（不缓存全量结果）**

```python
if resolved_return_iterator:
    return _iter_back2raw_results(
        final_tasks=final_tasks,
        worker_args_base=worker_args_base,
        num_workers=num_workers,
        ordered=ordered,
    )
```

```python
def _iter_back2raw_results(final_tasks, worker_args_base, num_workers, ordered=True):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_image_task, task, worker_args_base) for task in final_tasks]
        if ordered:
            for task, future in zip(final_tasks, futures):
                img_results = future.result()
                for roi_id, gtiff in img_results.items():
                    yield roi_id, task["img_id"], gtiff
            return

        future_map = {f: t["img_id"] for f, t in zip(futures, final_tasks)}
        for future in concurrent.futures.as_completed(future_map):
            img_id = future_map[future]
            img_results = future.result()
            for roi_id, gtiff in img_results.items():
                yield roi_id, img_id, gtiff
```

- [ ] **Step 4: 运行 tests 确认通过**

Run: `uv run pytest tests/test_geotiff.py -k "return_iterator_type or iterator_ordered_true"`

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/geotiff.py tests/test_geotiff.py
git commit -m "feat(geotiff): add ordered iterator return mode for streaming results"
```

### Task 4: 文档与示例更新

**Files:**

- Modify: `src/easyidp/geotiff.py` (docstring)

- [ ] **Step 1: 更新 docstring 参数说明与风险提示**

```python
return_iterator : bool, default False
return_iterator : bool | None, default None
    If True, force iterator yielding (roi_id, img_id, GeoTiff).
    If False, force dict return and never auto-switch.
    If None, auto mode: switch to iterator when task count is large.
    Note: iterator results can only be consumed once.

ordered : bool, default True
    Keep deterministic output order in iterator mode.
```

- [ ] **Step 2: 增加示例代码（保存即释放）**

```python
for roi_id, img_id, gtiff in idp.geotiff.back2raw2geotiff(..., return_iterator=True):
    gtiff.save(out_dir / f"grid_{roi_id}_{img_id}.tif", overwrite=True)
```

- [ ] **Step 3: 运行目标测试回归**

Run: `uv run pytest tests/test_geotiff.py -k "back2raw2geotiff"`

- [ ] **Step 4: Commit**

```bash
git add src/easyidp/geotiff.py docs/changelog.md
git commit -m "docs(geotiff): document iterator mode and memory warning behavior"
```

### Task 5: 全量验证

**Files:**

- Test: `tests/test_geotiff.py`

- [ ] **Step 1: 运行 geotiff 全测试**

Run: `uv run pytest tests/test_geotiff.py`

- [ ] **Step 2: 结果检查**

Expected: all pass, no new warnings/errors unrelated to known flaky cases.

- [ ] **Step 3: 最终提交**

```bash
git add src/easyidp/geotiff.py tests/test_geotiff.py
git commit -m "feat(geotiff): add memory guard warning and iterator return mode"
```

## 备注

- warning + 自动切换触发规则（简化版）：
  - `return_iterator is None`
  - `num_workers is None`
  - `sum(len(v) for v in back2raw_result.values()) > 2000`
- 用户显式 `return_iterator=False` 时，不自动切换 iterator。
- warning 文案按确认版本：
  - `数据量大且超过系统资源占用。either手动减少num_workers（默认为x）或使用return_iterator。`
- 默认行为更新为自动模式：`return_iterator=None`。
- iterator 模式需要在文档明确“只能消费一次”。
