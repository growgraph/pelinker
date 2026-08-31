"""pelinker -- property/entity linking for BERT-like models.

Importing torch's triton runtime eagerly, before anything can pull in TensorFlow, is a
workaround for a native-library conflict: if TensorFlow is initialised first, importing
``triton.runtime`` afterwards segfaults the interpreter (inside ``triton/knobs.py``).
pelinker loads both -- torch for the encoders, TensorFlow via ``tf-keras`` for
ParametricUMAP -- so entry points that touched them in that order died on import.

Both imports are optional: a torch build without CUDA ships no triton, and neither is
needed by the pure-dataclass modules.
"""

try:  # pragma: no cover - depends on the installed torch build
    import torch  # noqa: F401
    import triton.runtime  # noqa: F401
except ImportError:  # pragma: no cover
    pass
