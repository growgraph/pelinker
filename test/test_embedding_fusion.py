import numpy as np
import pandas as pd

from pelinker.embedding_fusion import (
    JOIN_KEYS,
    dedupe_mean_embed_by_keys,
    fused_property_vectors_from_paths,
    mention_level_concat_frames,
)


def test_mention_level_concat_inner_join():
    df1 = pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "property": ["a", "b"],
            "mention": ["x", "y"],
            "embed": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    df2 = pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "property": ["a", "b"],
            "mention": ["x", "z"],
            "embed": [[2.0, 2.0], [3.0, 3.0]],
        }
    )
    out = mention_level_concat_frames([df1, df2])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["property"] == "a"
    assert row["mention"] == "x"
    np.testing.assert_array_equal(
        row["embed"], np.array([1.0, 0.0, 2.0, 2.0], dtype=np.float32)
    )


def test_mention_level_dupes_averaged_within_source():
    df1 = pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "property": ["a", "a"],
            "mention": ["x", "x"],
            "embed": [[0.0, 2.0], [2.0, 0.0]],
        }
    )
    df2 = pd.DataFrame(
        {
            "pmid": ["1"],
            "property": ["a"],
            "mention": ["x"],
            "embed": [[1.0, 1.0]],
        }
    )
    out = mention_level_concat_frames([df1, df2])
    assert len(out) == 1
    np.testing.assert_allclose(out.iloc[0]["embed"], [1.0, 1.0, 1.0, 1.0], rtol=1e-5)


def test_mention_level_single_frame_rename():
    df = pd.DataFrame(
        {
            "pmid": ["1"],
            "property": ["p"],
            "mention": ["m"],
            "embed": [[1.0, 2.0]],
        }
    )
    out = mention_level_concat_frames([df])
    assert list(out.columns) == list(JOIN_KEYS) + ["embed"]
    np.testing.assert_array_equal(out.iloc[0]["embed"], np.array([1.0, 2.0]))


def test_dedupe_mean_embed():
    df = pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "property": ["a", "a"],
            "mention": ["z", "z"],
            "embed": [[1.0, 0.0], [0.0, 3.0]],
        }
    )
    out = dedupe_mean_embed_by_keys(df, keys=JOIN_KEYS)
    assert len(out) == 1
    np.testing.assert_array_equal(out.iloc[0]["embed"], np.array([0.5, 1.5]))


def test_property_fused_vectors_from_parquet_paths(tmp_path):
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    pd.DataFrame(
        {
            "pmid": ["1"],
            "property": ["foo"],
            "mention": ["m"],
            "embed": [[1.0, 0.0]],
        }
    ).to_parquet(p1)
    pd.DataFrame(
        {
            "pmid": ["1"],
            "property": ["foo"],
            "mention": ["m"],
            "embed": [[0.0, 2.0]],
        }
    ).to_parquet(p2)
    fused = fused_property_vectors_from_paths([p1, p2], kb_labels=None)
    assert "foo" in fused
    np.testing.assert_array_equal(fused["foo"], np.array([1.0, 0.0, 0.0, 2.0]))


def test_property_fused_intersection(tmp_path):
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "property": ["only_a", "both"],
            "mention": ["m", "m"],
            "embed": [[1.0], [3.0]],
        }
    ).to_parquet(p1)
    pd.DataFrame(
        {
            "pmid": ["1"],
            "property": ["both"],
            "mention": ["m"],
            "embed": [[4.0]],
        }
    ).to_parquet(p2)
    fused = fused_property_vectors_from_paths([p1, p2], kb_labels=None)
    assert set(fused.keys()) == {"both"}
    np.testing.assert_array_equal(fused["both"], np.array([3.0, 4.0]))
