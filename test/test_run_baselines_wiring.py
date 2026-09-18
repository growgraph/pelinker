"""Wiring of the fitted-linker system in the baseline runner.

`Linker.predict` takes `threshold`; the runner called it with `thr_score`, so the
`linker` system raised `TypeError` on its first call and had never executed. Only
`--systems lexical,encoder` had ever been run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run" / "eval" / "run_baselines.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_baselines", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rb = _load_module()


class RecordingLinker:
    """Accepts only the kwargs `Linker.predict` really has."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, texts, threshold=0.0, **kwargs):
        self.calls.append({"n_texts": len(texts), "threshold": threshold, **kwargs})

        class _Result:
            entities = [
                {"itext": 0, "a": 0, "b": 4, "entity_id_predicted": "kb::C0001"}
            ]

        return _Result()


def test_predict_receives_threshold_not_thr_score() -> None:
    linker = RecordingLinker()

    rows = rb._linker_predict_fn(linker, 0.25)(["some text", "other text"])

    assert linker.calls[0]["threshold"] == 0.25
    assert "thr_score" not in linker.calls[0]
    assert rows and rows[0]["entity_id_predicted"] == "kb::C0001"


def test_one_predict_call_covers_the_whole_batch() -> None:
    linker = RecordingLinker()

    rb._linker_predict_fn(linker, 0.0)(["a", "b", "c"])

    assert len(linker.calls) == 1
    assert linker.calls[0]["n_texts"] == 3


def test_span_link_fn_also_uses_threshold() -> None:
    linker = RecordingLinker()

    out = rb._linker_link_fn(linker, 0.5)("some text", 0, 4)

    assert linker.calls[0]["threshold"] == 0.5
    assert out == "kb::C0001"
