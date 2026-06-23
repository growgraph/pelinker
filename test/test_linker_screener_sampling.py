"""Screener training subsample for :meth:`~pelinker.model.Linker.fit`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.config import LinkerFitConfig
from pelinker.model import _screener_training_frame
from pelinker.onto import NEGATIVE_LABEL


def _synthetic_frame(n_kb: int, n_neg: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i in range(n_kb):
        rows.append(
            {
                "pmid": i,
                "entity": "kb_a",
                "mention": f"k{i}",
                "embed": rng.standard_normal(4).astype(np.float32),
            }
        )
    for j in range(n_neg):
        rows.append(
            {
                "pmid": 10_000 + j,
                "entity": NEGATIVE_LABEL,
                "mention": f"n{j}",
                "embed": rng.standard_normal(4).astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def test_screener_training_frame_unchanged_when_under_cap() -> None:
    frame = _synthetic_frame(n_kb=50, n_neg=20)
    cfg = LinkerFitConfig(screener_max_rows=100_000)
    out = _screener_training_frame(frame, cfg, negative_label=NEGATIVE_LABEL)
    assert out is frame
    assert len(out) == 70


def test_screener_training_frame_caps_and_preserves_classes() -> None:
    frame = _synthetic_frame(n_kb=5_000, n_neg=2_000)
    cfg = LinkerFitConfig(screener_max_rows=500, screener_seed=3)
    out = _screener_training_frame(frame, cfg, negative_label=NEGATIVE_LABEL)
    assert len(out) == 500
    assert (out["entity"] == NEGATIVE_LABEL).any()
    assert (out["entity"] != NEGATIVE_LABEL).any()


def test_screener_training_frame_deterministic() -> None:
    frame = _synthetic_frame(n_kb=3_000, n_neg=1_000)
    cfg = LinkerFitConfig(screener_max_rows=400, screener_seed=9)
    a = _screener_training_frame(frame, cfg, negative_label=NEGATIVE_LABEL)
    b = _screener_training_frame(frame, cfg, negative_label=NEGATIVE_LABEL)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
