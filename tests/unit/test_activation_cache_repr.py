"""Regression tests pinning ActivationCache repr/display behavior (#1126).

Verifies that formatting an ``ActivationCache`` via ``repr``, ``str``, ``pprint``, or IPython's
pretty-printer only inspects key names and never reads or formats the underlying cached tensors.
"""

from __future__ import annotations

import pprint
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache


class _KeysOnlyDict(dict):
    """Dict subclass that forbids value/item reads once armed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.armed = False

    def __getitem__(self, key: Any) -> Any:
        if self.armed:
            raise AssertionError(f"ActivationCache repr accessed cache_dict[{key!r}]")
        return super().__getitem__(key)

    def values(self) -> Any:
        if self.armed:
            raise AssertionError("ActivationCache repr accessed cache_dict.values()")
        return super().values()

    def items(self) -> Any:
        if self.armed:
            raise AssertionError("ActivationCache repr accessed cache_dict.items()")
        return super().items()

    def get(self, key: Any, default: Any = None) -> Any:
        if self.armed:
            raise AssertionError(f"ActivationCache repr called cache_dict.get({key!r})")
        return super().get(key, default)


class _NoTouchTensor(torch.Tensor):
    """Real torch.Tensor subclass that fails if formatted or operated on while armed."""

    _armed = False

    def __repr__(self) -> str:
        if type(self)._armed:
            raise AssertionError("ActivationCache repr triggered tensor.__repr__()")
        return super().__repr__()

    def __str__(self) -> str:
        if type(self)._armed:
            raise AssertionError("ActivationCache repr triggered tensor.__str__()")
        return super().__str__()

    def __format__(self, format_spec: str) -> str:
        if type(self)._armed:
            raise AssertionError("ActivationCache repr triggered tensor.__format__()")
        return super().__format__(format_spec)

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any:
        if cls._armed:
            raise AssertionError(f"ActivationCache repr invoked torch op {func!r} on cached tensor")
        return super().__torch_function__(func, types, args, kwargs)


@contextmanager
def _arm_guards(cache_dict: Optional[_KeysOnlyDict] = None) -> Iterator[None]:
    """Arm dict and tensor guards after __init__ (so beartype checks pass cleanly)."""
    if cache_dict is not None:
        cache_dict.armed = True
    _NoTouchTensor._armed = True
    try:
        yield
    finally:
        _NoTouchTensor._armed = False
        if cache_dict is not None:
            cache_dict.armed = False


def test_repr_and_str_format_keys_only() -> None:
    cache = ActivationCache(
        {
            "hook_embed": torch.zeros((1, 4, 8)),
            "blocks.0.hook_resid_pre": torch.ones((1, 4, 8)),
        },
        model=None,
    )
    expected = "ActivationCache with keys ['hook_embed', 'blocks.0.hook_resid_pre']"
    assert repr(cache) == expected
    assert str(cache) == expected

    empty_cache = ActivationCache({}, model=None)
    assert repr(empty_cache) == "ActivationCache with keys []"
    assert str(empty_cache) == "ActivationCache with keys []"


def test_repr_str_and_pformat_do_not_access_cache_dict_values() -> None:
    cache_dict = _KeysOnlyDict(
        {
            "hook_embed": torch.zeros((2, 4, 8)),
            "blocks.0.attn.hook_pattern": torch.zeros((2, 2, 4, 4)),
        }
    )
    cache = ActivationCache(cache_dict, model=None)
    expected = "ActivationCache with keys ['hook_embed', 'blocks.0.attn.hook_pattern']"

    with _arm_guards(cache_dict):
        assert repr(cache) == expected
        assert str(cache) == expected
        assert pprint.pformat(cache) == expected


def test_repr_str_and_pformat_do_not_touch_tensors() -> None:
    cache_dict = _KeysOnlyDict(
        {
            "hook_embed": torch.zeros((2, 4, 8)).as_subclass(_NoTouchTensor),
            "blocks.0.hook_resid_post": torch.zeros((2, 4, 8)).as_subclass(_NoTouchTensor),
        }
    )
    cache = ActivationCache(cache_dict, model=None)
    expected = "ActivationCache with keys ['hook_embed', 'blocks.0.hook_resid_post']"

    with _arm_guards(cache_dict):
        assert repr(cache) == expected
        assert str(cache) == expected
        assert pprint.pformat(cache) == expected


def test_no_custom_notebook_display_hooks() -> None:
    assert "__str__" not in ActivationCache.__dict__
    for hook_name in ("_repr_html_", "_repr_pretty_", "_repr_mimebundle_"):
        assert not hasattr(ActivationCache, hook_name)


def test_ipython_pretty_matches_repr_without_touching_tensors() -> None:
    pretty_mod = pytest.importorskip("IPython.lib.pretty")
    cache_dict = _KeysOnlyDict(
        {
            "hook_embed": torch.zeros((2, 4, 8)).as_subclass(_NoTouchTensor),
            "blocks.0.hook_resid_pre": torch.zeros((2, 4, 8)).as_subclass(_NoTouchTensor),
        }
    )
    cache = ActivationCache(cache_dict, model=None)
    expected = "ActivationCache with keys ['hook_embed', 'blocks.0.hook_resid_pre']"

    with _arm_guards(cache_dict):
        assert pretty_mod.pretty(cache) == expected
