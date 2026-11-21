"""Probe ActivationCache repr/display cost with fixed keys and growing tensor sizes.

Imports the real ``ActivationCache`` from this checkout (not a stub) and compares:
- ``repr(cache)``, ``str(cache)``, and IPython ``pretty(cache)`` on ``ActivationCache``
- ``repr(cache.cache_dict)`` on the underlying raw ``dict[str, torch.Tensor]``

Examples:
    python debugging/activation_cache_repr_probe.py
    python debugging/activation_cache_repr_probe.py --device cuda --layers 32
    python debugging/activation_cache_repr_probe.py --shapes 1x4x8 2x128x768 4x256x1024
"""

from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Prefer this checkout over any globally installed transformer_lens package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from transformer_lens.ActivationCache import ActivationCache  # noqa: E402

try:
    from IPython.lib.pretty import pretty as ipython_pretty
except ImportError:
    ipython_pretty = None

PER_LAYER_HOOKS: Tuple[str, ...] = (
    "hook_resid_pre",
    "ln1.hook_scale",
    "ln1.hook_normalized",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_rot_q",
    "attn.hook_rot_k",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    "attn.hook_z",
    "attn.hook_result",
    "hook_attn_out",
    "hook_resid_mid",
    "ln2.hook_scale",
    "ln2.hook_normalized",
    "mlp.hook_pre",
    "mlp.hook_pre_linear",
    "mlp.hook_post",
    "hook_mlp_out",
    "hook_resid_post",
)

DEFAULT_SHAPES: Tuple[Tuple[int, ...], ...] = (
    (1, 4, 8),
    (1, 32, 256),
    (2, 128, 768),
    (4, 256, 1024),
)

DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_hook_keys(n_layers: int) -> List[str]:
    """Build a fixed HookedTransformer-style key list for ``n_layers`` blocks."""
    keys = ["hook_embed", "hook_pos_embed"]
    for layer in range(n_layers):
        for hook_name in PER_LAYER_HOOKS:
            keys.append(f"blocks.{layer}.{hook_name}")
    keys.extend(["ln_final.hook_scale", "ln_final.hook_normalized"])
    return keys


def make_cache(
    keys: List[str],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> ActivationCache:
    """Allocate a real ActivationCache with distinct tensors for each key."""
    cache_dict = {key: torch.zeros(shape, dtype=dtype, device=device) for key in keys}
    return ActivationCache(cache_dict, model=None, has_batch_dim=True)


def cache_tensor_bytes(cache: ActivationCache) -> int:
    return sum(t.nelement() * t.element_size() for t in cache.cache_dict.values())


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def time_fn(fn: Callable[[], str], device: torch.device, repeats: int) -> Tuple[float, int]:
    """Return ``(median_ms, output_char_len)`` with device sync around each call."""
    sync_device(device)
    out = fn()
    sync_device(device)
    out_len = len(out)

    times_ms: List[float] = []
    for _ in range(repeats):
        sync_device(device)
        t0 = time.perf_counter()
        out = fn()
        sync_device(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], out_len


def parse_shapes(raw_shapes: Optional[List[str]]) -> List[Tuple[int, ...]]:
    if not raw_shapes:
        return list(DEFAULT_SHAPES)
    parsed: List[Tuple[int, ...]] = []
    for spec in raw_shapes:
        dims = tuple(int(part.strip()) for part in spec.split("x"))
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"Invalid shape {spec!r}; expected e.g. 2x128x768")
        parsed.append(dims)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe real ActivationCache repr/display with fixed keys and bigger tensors."
    )
    parser.add_argument("--layers", type=int, default=12, help="Number of layers (default: 12)")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="Tensor shapes as DxDxD (e.g. 1x4x8 1x32x256 2x128x768 4x256x1024)",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="float32")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per case (default: 5)")
    parser.add_argument("--skip-dict-repr", action="store_true", help="Skip repr(cache.cache_dict)")
    parser.add_argument(
        "--print-threshold",
        type=int,
        default=None,
        help="Optional torch.set_printoptions(threshold=...) override",
    )
    args = parser.parse_args()

    if args.print_threshold is not None:
        torch.set_printoptions(threshold=args.print_threshold)

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    shapes = parse_shapes(args.shapes)
    keys = build_hook_keys(args.layers)

    print(f"ActivationCache source : {inspect.getsourcefile(ActivationCache)}")
    print(f"torch version          : {torch.__version__} (device={device}, dtype={dtype})")
    print(f"IPython pretty         : {'available' if ipython_pretty is not None else 'not installed'}")
    print(f"Fixed key count        : {len(keys)} ({args.layers} layers)")
    print()

    header = (
        f"{'shape':<14} | {'tensors MB':>10} | {'repr ms':>8} | {'str ms':>8} | "
        f"{'pretty ms':>9} | {'repr chars':>10} | {'dict ms':>9} | {'dict chars':>10}"
    )
    print(header)
    print("-" * len(header))

    for shape in shapes:
        cache = make_cache(keys, shape=shape, dtype=dtype, device=device)
        mem_mb = cache_tensor_bytes(cache) / (1024.0 * 1024.0)

        repr_ms, repr_len = time_fn(lambda: repr(cache), device, args.repeats)
        str_ms, _ = time_fn(lambda: str(cache), device, args.repeats)
        if ipython_pretty is not None:
            pretty_ms, _ = time_fn(lambda: ipython_pretty(cache), device, args.repeats)
            pretty_col = f"{pretty_ms:9.3f}"
        else:
            pretty_col = f"{'n/a':>9}"

        if args.skip_dict_repr:
            dict_ms_col, dict_len_col = f"{'skipped':>9}", f"{'skipped':>10}"
        else:
            dict_ms, dict_len = time_fn(lambda: repr(cache.cache_dict), device, args.repeats)
            dict_ms_col, dict_len_col = f"{dict_ms:9.3f}", f"{dict_len:10d}"

        shape_str = "x".join(str(d) for d in shape)
        print(
            f"{shape_str:<14} | {mem_mb:10.2f} | {repr_ms:8.3f} | {str_ms:8.3f} | "
            f"{pretty_col} | {repr_len:10d} | {dict_ms_col} | {dict_len_col}"
        )

        del cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
