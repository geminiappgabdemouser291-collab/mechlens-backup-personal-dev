# Triage: `ActivationCache.__repr__` Performance (#1126)

* **Branch:** `investigate/activation-cache-repr-1126` (branched off `33387b7`)
* **Related files on branch:**
  * [`debugging/activation_cache_repr_probe.py`](../debugging/activation_cache_repr_probe.py)
  * [`tests/unit/test_activation_cache_repr.py`](../tests/unit/test_activation_cache_repr.py)

## Verdict: Is `ActivationCache.__repr__` Itself the Problem?

**No** — `ActivationCache.__repr__` is not stalling Python on large cached tensors.

* **Tensor size does not affect `repr(cache)`:** Scaling 256 cached tensors from `0.03 MB` to `1,024 MB` leaves `repr(cache)` flat at `0.024–0.025 ms` (`0.038–0.043 ms` via `IPython.lib.pretty`), because `__repr__` never reads tensor values or triggers device-to-host sync.
* **Key count stays sub-millisecond in Python:** Even at 256 layers (`5,380` keys), `repr(cache)` takes `0.481 ms` (`0.562 ms` via `IPython.lib.pretty`), compared to `1,143.2 ms` for formatting the raw `cache.cache_dict`.
* **Only remaining caveat in `__repr__`:** Output length is untruncated on a single line (`7,099` chars at 12 layers, `76,045` chars at 126 layers, `156,775` chars at 256 layers). For very deep models, rendering that single unbroken line can still make a notebook frontend sluggish — so truncating long key lists is a reasonable UX improvement if confirmed, not a fix for slow tensor formatting.

## What Issue #1126 Reports

* **Title:** `bug: ActivationCache repr is very slow for large caches (#1126)`
* **Report:** Printing a large cache hangs the notebook, and asks whether `repr` could be truncated.
* **Not specified in the report:** Whether the printed object is an `ActivationCache` instance or a raw `dict[str, torch.Tensor]` (such as `cache.cache_dict` or the dict returned by `run_with_cache(..., return_cache_object=False)`), model depth / key count, tensor shapes, device (CPU vs. GPU), or notebook frontend.

## What the Code Actually Does

### 1. `ActivationCache.__repr__` formats keys only

In `transformer_lens/ActivationCache.py` (lines 176–183):

```python
def __repr__(self) -> str:
    return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

* It only materializes `list(self.cache_dict.keys())` and formats the list of key names.
* It never reads `self.cache_dict.values()`, never invokes `torch.Tensor.__repr__`, and does not inspect tensor shapes, dtypes, or values.
* Because cached tensors are never accessed, `repr(cache)` does not trigger CUDA or MPS device-to-host synchronization.

### 2. `str()`, `pprint`, and notebook (`IPython`) display paths

* `ActivationCache` does not override `__str__`, so `print(cache)` delegates directly to `ActivationCache.__repr__`.
* `ActivationCache` does not subclass `dict` and defines no IPython display hooks (`_repr_pretty_`, `_repr_html_`, `_repr_mimebundle_`, etc.), so `pprint.pformat(cache)` and `IPython.lib.pretty.pretty(cache)` also fall back to `ActivationCache.__repr__` without reading tensor values.
* This keys-only, zero-tensor-touch behavior is pinned by `tests/unit/test_activation_cache_repr.py` (5/5 passing in CI run `#44917735528`).

### 3. Separate slow-`repr` workaround elsewhere in the repo

`transformer_lens/hook_points.py` (lines 248–254 and 273–276) guards against calling `partial.__repr__()` directly when naming wrapped forward and backward hooks:

```python
if isinstance(hook, partial):
    # partial.__repr__() can be extremely slow if arguments contain large objects, which
    # is common when caching tensors.
    full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
```

That guard addresses `functools.partial` hooks that bind large objects or raw cache dicts during hook registration, which is separate from `ActivationCache.__repr__`.

## Report vs. Observed Checkout Behavior

| Question | What #1126 implies | What this checkout actually does |
| --- | --- | --- |
| Does `repr(cache)` touch or format cached tensors? | Printing a "large cache" stalls on cache contents. | **No.** `ActivationCache.__repr__` formats only `list(self.cache_dict.keys())`. Growing tensors from `0.03 MB` to `1,024 MB` (fixed 256 keys) keeps `repr(cache)` flat at `0.024–0.025 ms` and `7,099` chars (`IPython.lib.pretty`: `0.038–0.043 ms`). |
| What scales `repr(cache)` cost? | Cache size broadly. | **Key count only** (linear in number of keys). Growing from 12 layers (256 keys) to 256 layers (5,380 keys) moves `repr(cache)` from `0.025 ms` to `0.481 ms` (`IPython.lib.pretty`: `0.562 ms`), while the output string grows from `7,099` to `156,775` characters on a single line. |
| Where does slow tensor formatting actually occur? | Attributed to `ActivationCache` repr. | **Raw `dict` repr (`cache.cache_dict` / `dict(cache)` / `return_cache_object=False`).** Formatting `cache.cache_dict` hits `torch.Tensor.__repr__` on every entry: `139.3 ms` (`275,617` chars) at 256 keys / 1 GB, and `1,143.2 ms` (`1,571,689` chars) at 5,380 keys on CPU even with PyTorch's default truncation. |

## Measured CPU Probe Numbers

Measured with `debugging/activation_cache_repr_probe.py` against `transformer_lens/ActivationCache.py` in CI run `#44917735528` (`torch 2.10.0+cu128`, `device=cpu`, `dtype=torch.float32`, `IPython pretty: available`, 5 timed repeats, median ms):

### Sweep 1: Fixed keys (256 keys, 12 layers), growing tensors

| Shape | Tensors MB | `repr` ms | `str` ms | `pretty` ms | `repr` chars | `dict` ms | `dict` chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1x4x8` | 0.03 | 0.024 | 0.023 | 0.043 | 7,099 | 25.633 | 51,873 |
| `1x32x256` | 8.00 | 0.025 | 0.023 | 0.043 | 7,099 | 54.257 | 74,401 |
| `2x128x768` | 192.00 | 0.024 | 0.024 | 0.038 | 7,099 | 82.647 | 141,473 |
| `4x256x1024` | 1024.00 | 0.024 | 0.023 | 0.038 | 7,099 | 139.304 | 275,617 |

### Sweep 2: Fixed tensor size (`1x32x256`), growing key counts

| Layers | Keys | Tensors MB | `repr` ms | `str` ms | `pretty` ms | `repr` chars | `dict` ms | `dict` chars |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 256 | 8.00 | 0.025 | 0.024 | 0.037 | 7,099 | 54.438 | 74,401 |
| 32 | 676 | 21.12 | 0.061 | 0.061 | 0.083 | 19,099 | 142.512 | 196,861 |
| 80 | 1,684 | 52.62 | 0.149 | 0.148 | 0.201 | 47,899 | 354.949 | 490,765 |
| 126 | 2,650 | 82.81 | 0.236 | 0.231 | 0.289 | 76,045 | 560.001 | 772,969 |
| 256 | 5,380 | 168.12 | 0.481 | 0.466 | 0.562 | 156,775 | 1,143.227 | 1,571,689 |

## Open Hypotheses Still Worth Checking With the Reporter

Because `ActivationCache.__repr__` stays sub-millisecond in Python even at 1 GB of tensors or 5,380 keys, a real notebook hang most likely comes from one of:

1. **Printing the raw `dict` instead of `ActivationCache`** (e.g., `cache.cache_dict`, `dict(cache)`, or `run_with_cache(..., return_cache_object=False)`), especially on CUDA where `torch.Tensor.__repr__` forces host sync per tensor, or with a raised `torch.set_printoptions(threshold=...)`.
2. **Notebook frontend rendering of a huge single-line key list** when a model has thousands of keys (e.g., `156,775` characters on one unbroken line at 5,380 keys, where Python formatting is `0.48 ms` but browser or VS Code output rendering can still lag).
3. **Something outside `ActivationCache.__repr__`** (an earlier un-synced GPU op, a notebook variable inspector traversing `cache.cache_dict`, or a different/older version).

## Draft Reply for #1126 (Not Posted)

> Thanks for filing this! I checked `ActivationCache.__repr__` on `main` and benchmarked it across tensor sizes up to 1 GB and key counts up to 5,380 keys — right now it only formats `list(self.cache_dict.keys())` without reading any tensor values, and stays under `0.5 ms` in Python (including via `IPython.lib.pretty`), whereas formatting the raw `cache.cache_dict` takes over `1 s` at 5,380 keys on CPU.
>
> To pin down what's hanging on your end (and whether truncating long key lists in `__repr__` would actually fix it), could you share a few details?
>
> 1. What is `type(cache)` for the object being printed — an `ActivationCache` instance, or a raw `dict` (e.g. `cache.cache_dict` or `run_with_cache(..., return_cache_object=False)`)?
> 2. Roughly how many keys (`len(cache)`), which model, and what device (`cpu` / `cuda` / `mps`)?
> 3. The exact snippet that hangs (`print(cache)`, bare `cache` in a cell, `pprint`, variable inspector, etc.) and whether `torch.set_printoptions(...)` was modified?
> 4. Which notebook frontend (JupyterLab, VS Code, Colab, etc.) and your TransformerLens + PyTorch versions?
> 5. If you time `s = repr(cache)` without displaying `s`, is Python itself slow, or does the hang only happen when rendering the output in the notebook?
