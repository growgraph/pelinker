"""pelinker -- property/entity linking for BERT-like models.

Deliberately free of heavy imports: importing anything under ``pelinker`` must not pull in
torch or TensorFlow. Scripts that only touch the KB, the gold schema or the evaluation
harness pay nothing for the ML stack, and a process that has already loaded TensorFlow
(a debugger, a notebook) can import this package without tripping the native
torch/TensorFlow load-order conflict.

That conflict is real, and it is handled where it arises:
:func:`pelinker.core.runtime.preload_torch_before_tensorflow`, called at the two sites
that import ``umap.parametric_umap``.
"""
