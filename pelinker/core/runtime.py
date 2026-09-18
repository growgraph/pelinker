"""Native-library load order for the one place it matters.

If TensorFlow is initialised first, importing ``triton.runtime`` afterwards segfaults the
interpreter inside ``triton/knobs.py``. pelinker loads both — torch for the encoders,
TensorFlow via ``tf-keras`` for ParametricUMAP — so the torch side must come first.

This used to live in ``pelinker/__init__.py``, which made *every* import of anything under
``pelinker`` drag in torch: over a thousand modules and seconds of startup for scripts
that only read a CSV and call an HTTP API, and a hard crash under a debugger that had
already loaded TensorFlow. TensorFlow only ever enters through
``umap.parametric_umap``, so the ordering is enforced at those call sites instead.

Call :func:`preload_torch_before_tensorflow` immediately before importing anything that
pulls in TensorFlow.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PRELOADED = False


def preload_torch_before_tensorflow() -> bool:
    """Import torch (and its triton runtime) ahead of TensorFlow. Idempotent.

    Returns True when torch is present, False when it is not installed — a torch build
    without CUDA ships no triton, and neither is needed by the pure-dataclass modules, so
    absence is not an error.
    """
    global _PRELOADED
    if _PRELOADED:
        return True
    try:
        import torch  # noqa: F401
        import triton.runtime  # noqa: F401
    except ImportError:
        logger.debug("torch/triton unavailable; skipping the pre-TensorFlow import")
        return False
    _PRELOADED = True
    return True
