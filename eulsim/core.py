"""The fastest available in-place simulator: the compiled ``_core.CoreSim`` when it has been
built (``python setup_core.py build_ext --inplace``), else the pure-Python ``sim.EulerSim``.
Both have the same API and make the same choices; set ``EULSIM_PURE=1`` to force the latter."""
from __future__ import annotations

import os

from .sim import EulerSim

try:
    if os.environ.get("EULSIM_PURE"):
        raise ImportError("EULSIM_PURE set")
    from ._core import CoreSim as FastSim
    COMPILED = True
except ImportError:
    FastSim = EulerSim
    COMPILED = False

__all__ = ["COMPILED", "EulerSim", "FastSim"]
