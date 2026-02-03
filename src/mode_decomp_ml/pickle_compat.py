"""Compatibility helpers for loading legacy pickle artifacts."""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_pickle_compat(path: str | Path) -> Any:
    """Load pickle artifacts with numpy BitGenerator fallback."""
    try:
        return _load_pickle(path)
    except ValueError as exc:
        message = str(exc)
        if "BitGenerator module" not in message and "legacy MT19937 state" not in message:
            raise

    try:
        import numpy.random._pickle as np_pickle
        import numpy as np
    except Exception:
        raise

    original_bitgen_ctor = np_pickle.__bit_generator_ctor
    original_randomstate_ctor = np_pickle.__randomstate_ctor
    original_generator_ctor = np_pickle.__generator_ctor

    def _compat_ctor(bit_generator_name: Any = "MT19937"):
        if isinstance(bit_generator_name, np.random.BitGenerator):
            return bit_generator_name
        if isinstance(bit_generator_name, type):
            name = bit_generator_name.__name__
        else:
            name = bit_generator_name
        if name in np_pickle.BitGenerators:
            return np_pickle.BitGenerators[name]()
        if isinstance(bit_generator_name, type):
            return bit_generator_name()
        return original_bitgen_ctor(bit_generator_name)

    def _compat_randomstate_ctor(bit_generator_name: Any = "MT19937", bit_generator_ctor: Any = None):
        return np_pickle.RandomState(_compat_ctor(bit_generator_name))

    def _compat_generator_ctor(bit_generator_name: Any = "MT19937", bit_generator_ctor: Any = None):
        return np.random.Generator(_compat_ctor(bit_generator_name))

    class _CompatUnpickler(pickle._Unpickler):
        def load_build(self):  # type: ignore[override]
            stack = self.stack
            state = stack.pop()
            inst = stack[-1]
            setstate = getattr(inst, "__setstate__", None)
            if setstate is not None:
                try:
                    setstate(state)
                except ValueError as exc:
                    if isinstance(inst, np.random.MT19937) and "legacy MT19937 state" in str(exc):
                        return
                    raise
                return
            slotstate = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if state:
                inst_dict = inst.__dict__
                intern = __import__("sys").intern
                for k, v in state.items():
                    if type(k) is str:
                        inst_dict[intern(k)] = v
                    else:
                        inst_dict[k] = v
            if slotstate:
                for k, v in slotstate.items():
                    setattr(inst, k, v)

    _CompatUnpickler.dispatch = pickle._Unpickler.dispatch.copy()
    _CompatUnpickler.dispatch[pickle.BUILD[0]] = _CompatUnpickler.load_build

    np_pickle.__bit_generator_ctor = _compat_ctor
    np_pickle.__randomstate_ctor = _compat_randomstate_ctor
    np_pickle.__generator_ctor = _compat_generator_ctor
    try:
        with Path(path).open("rb") as fh:
            return _CompatUnpickler(fh).load()
    finally:
        np_pickle.__bit_generator_ctor = original_bitgen_ctor
        np_pickle.__randomstate_ctor = original_randomstate_ctor
        np_pickle.__generator_ctor = original_generator_ctor
