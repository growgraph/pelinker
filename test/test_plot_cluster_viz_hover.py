"""Unit tests for cluster-viz hover / legend formatting helpers."""

from __future__ import annotations

import pathlib

import pandas as pd

from pelinker.plotting import (
    _format_hover_float,
    _truncate_legend_label,
    _wrap_hover_html,
    plot_cluster_viz,
)


def test_truncate_legend_label_short_unchanged() -> None:
    assert _truncate_legend_label("alpha") == "alpha"


def test_truncate_legend_label_long() -> None:
    long = "alpha-67--beta-33--gamma-15--delta-10"
    out = _truncate_legend_label(long, max_chars=20)
    assert len(out) <= 20
    assert out.endswith("…")


def test_wrap_hover_html_inserts_breaks() -> None:
    text = " ".join(["word"] * 30)
    wrapped = _wrap_hover_html(text, width=20)
    assert "<br>" in wrapped
    assert "word" in wrapped


def test_format_hover_float_three_decimals() -> None:
    assert _format_hover_float(0.123456) == "0.123"
    assert _format_hover_float(None) == ""


def test_plot_cluster_viz_writes_html_with_legend_hint(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "class": [
                "alpha-67--beta-33--gamma-15",
                "alpha-67--beta-33--gamma-15",
                "other-90--noise-10",
            ],
            "cviz_00": [0.1, 0.2, 0.3],
            "cviz_01": [0.4, 0.5, 0.6],
            "cluster_score": [0.98765, 0.5, 0.12345],
            "context": [" ".join(["ctx"] * 40)] * 3,
        }
    )
    out = tmp_path / "viz.html"
    plot_cluster_viz(df, output_path=out, viz_method="pca")
    html = out.read_text(encoding="utf-8")
    assert "dbl-click isolate" in html
    assert "double-click to isolate" in html
    assert "drag to pan" in html
    # Legend title must stay single-line (multi-line HTML clips items in Plotly).
    assert "Cluster (dbl-click isolate)" in html
    assert "Cluster<br>" not in html
    assert '"dragmode":"pan"' in html or '"dragmode": "pan"' in html
