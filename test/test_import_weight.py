"""Import weight and the torch/TensorFlow load-order guarantee.

Two invariants, both learned from a crash: importing anything under ``pelinker`` must not
drag in the ML stack, and the one module that pulls TensorFlow must import torch first.
Violating either produces a SIGSEGV rather than an exception, so they are checked in
subprocesses.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

LIGHT_MODULES = ("torch", "triton", "tensorflow", "keras", "umap")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_package_root_pulls_in_nothing_heavy() -> None:
    result = _run(
        f"""
        import sys
        import pelinker
        heavy = [m for m in {LIGHT_MODULES!r} if m in sys.modules]
        assert not heavy, heavy
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_the_gold_pipeline_imports_stay_light() -> None:
    """A script that reads a CSV and calls an HTTP API must not import torch."""
    result = _run(
        f"""
        import sys
        from pelinker.eval.llm import complete
        from pelinker.eval.harness import load_gold_docs
        from pelinker.kb.ground_truth import GT_DIRECTIONS
        heavy = [m for m in {LIGHT_MODULES!r} if m in sys.modules]
        assert not heavy, heavy
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


@pytest.mark.heavy
def test_transform_import_survives_a_later_triton_import() -> None:
    """The documented segfault: TensorFlow first, then ``import triton.runtime``.

    ``pelinker.clustering.transform`` imports ``umap``, which loads TensorFlow, so it must
    preload torch/triton beforehand. Without that, this subprocess dies with SIGSEGV
    (-11) instead of failing an assertion.
    """
    result = _run(
        """
        import pelinker.clustering.transform  # noqa: F401
        import triton.runtime  # noqa: F401
        print("ok")
        """
    )

    if result.returncode != 0 and "No module named" in result.stderr:
        pytest.skip("torch/triton or umap not installed in this environment")
    assert result.returncode == 0, f"exit {result.returncode} (-11 is SIGSEGV)"
    assert "ok" in result.stdout


@pytest.mark.heavy
def test_gold_pipeline_imports_after_tensorflow_is_already_loaded() -> None:
    """The reported failure: a debugger had TensorFlow in the process first."""
    result = _run(
        """
        import tensorflow  # noqa: F401
        from pelinker.eval.llm import complete  # noqa: F401
        from pelinker.kb.ground_truth import GT_DIRECTIONS  # noqa: F401
        print("ok")
        """
    )

    if result.returncode != 0 and "No module named" in result.stderr:
        pytest.skip("tensorflow not installed in this environment")
    assert result.returncode == 0, f"exit {result.returncode} (-11 is SIGSEGV)"
    assert "ok" in result.stdout
