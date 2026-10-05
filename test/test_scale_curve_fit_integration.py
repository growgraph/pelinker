"""min_cluster_size resolution through a real Linker.fit (scale curve vs explicit)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pelinker.core.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    TransformConfig,
)
from pelinker.model import Linker
from pelinker.search.scale_curve import SCALE_CURVE_SCHEMA, load_scale_curve
from pelinker.core.scaling import ScaleRung, fit_scale_curve


def _mentions_parquet(tmp_path: Path, n_ent: int = 24) -> tuple[Path, dict[str, str]]:
    p = tmp_path / "mentions.parquet"
    rows = [
        {
            "pmid": pmid,
            "entity": f"p{k}",
            "mention": "m",
            "embed": [float(k), float(k) * 0.1, 1.0, 0.5],
        }
        for k in range(n_ent)
        for pmid in ("1", "2", "3", "4")
    ]
    pd.DataFrame(rows).to_parquet(p)
    return p, {f"e{k}": f"p{k}" for k in range(n_ent)}


def _curve():
    """A curve whose slope is high enough that N materially moves the answer."""
    return fit_scale_curve(
        [
            ScaleRung(100, 4, 0.5, 10.0, 2),
            ScaleRung(1_000, 8, 0.5, 10.0, 2),
            ScaleRung(10_000, 16, 0.5, 10.0, 2),
        ]
    )


def _fit(tmp_path: Path, *, min_cluster_size, scale_curve) -> Linker:
    parquet, labels_map = _mentions_parquet(tmp_path)
    linker = Linker(
        labels_map=labels_map,
        embedding_metadata=EmbeddingModelMetadata(
            sources=(EmbeddingSourceSpec(model_type="a", layers_spec="1"),)
        ),
    )
    linker.fit(
        parquet,
        transform_config=TransformConfig(
            pca_components=4,
            umap_components=2,
            cluster_viz_components=2,
            umap_seed=13,
        ),
        min_cluster_size=min_cluster_size,
        fit_config=LinkerFitConfig(
            batch_size=500,
            predict_mode="legacy",
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
            scale_curve=scale_curve,
        ),
    )
    return linker


def test_scale_curve_drives_min_cluster_size_when_none_given(tmp_path: Path) -> None:
    linker = _fit(tmp_path, min_cluster_size=None, scale_curve=_curve())

    prov = linker.min_cluster_size_provenance
    assert prov is not None
    assert prov.source == "scale_curve"
    # 96 manifold rows (24 entities x 4 pmids, no negatives) -> curve predicts ~4.
    assert prov.n_rows_realized == 96
    assert prov.extrapolation is not None
    assert prov.min_cluster_size == prov.extrapolation.min_cluster_size
    assert linker.clustering_fit_metrics is not None
    assert linker.clustering_fit_metrics.min_cluster_size == prov.min_cluster_size


def test_explicit_min_cluster_size_overrides_the_curve_but_records_it(
    tmp_path: Path,
) -> None:
    linker = _fit(tmp_path, min_cluster_size=3, scale_curve=_curve())

    prov = linker.min_cluster_size_provenance
    assert prov is not None
    assert prov.source == "explicit"
    assert prov.min_cluster_size == 3
    assert linker.clustering_fit_metrics.min_cluster_size == 3
    # The curve's own prediction is still recorded so the disagreement stays visible.
    assert prov.extrapolation is not None
    assert prov.extrapolation.min_cluster_size != 3


def test_without_a_curve_the_documented_default_is_used(tmp_path: Path) -> None:
    linker = _fit(tmp_path, min_cluster_size=None, scale_curve=None)

    prov = linker.min_cluster_size_provenance
    assert prov is not None
    assert prov.source == "default"
    assert prov.min_cluster_size == 20
    assert prov.extrapolation is None


def test_provenance_reaches_the_fit_report(tmp_path: Path) -> None:
    from pelinker.reports.io import clustering_report_to_jsonable_dict

    linker = _fit(tmp_path, min_cluster_size=None, scale_curve=_curve())
    report = linker.take_fit_clustering_report()
    assert report is not None

    payload = clustering_report_to_jsonable_dict(report)

    block = payload["min_cluster_size_provenance"]
    assert block is not None
    assert block["source"] == "scale_curve"
    assert block["n_rows_realized"] == 96
    assert payload["n_rows_realized"] == 96
    assert block["extrapolation"]["extrapolation_ratio"] == pytest.approx(96 / 10_000)
    # The whole report must stay JSON-serializable with the new block attached.
    json.dumps(payload)


def test_scale_curve_json_round_trips_through_the_loader(tmp_path: Path) -> None:
    curve = _curve()
    path = tmp_path / "scale_curve.json"
    path.write_text(
        json.dumps({"schema": SCALE_CURVE_SCHEMA, "curve": curve.to_jsonable()}),
        encoding="utf-8",
    )

    loaded = load_scale_curve(path)

    assert loaded.rungs == curve.rungs
    assert loaded.log_slope == pytest.approx(curve.log_slope)


def test_loader_rejects_an_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "scale_curve.json"
    path.write_text(
        json.dumps({"schema": "pelinker.search.scale_curve.v99", "curve": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected schema"):
        load_scale_curve(path)
