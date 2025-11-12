"""
read from resultant pq files:

from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from typing import Iterator

def read_parquet_exact_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(path)
    buf = []
    buf_rows = 0

    for batch in pf.iter_batches(batch_size=chunk_rows):
        df = batch.to_pandas(types_mapper=None)  # preserves list columns
        buf.append(df)
        buf_rows += len(df)

        while buf_rows >= chunk_rows:
            # assemble exactly chunk_rows
            out, used = [], 0
            while used < chunk_rows and buf:
                head = buf[0]
                take = min(chunk_rows - used, len(head))
                out.append(head.iloc[:take])
                used += take
                if take == len(head):
                    buf.pop(0)
                else:
                    buf[0] = head.iloc[take:]
            buf_rows -= chunk_rows
            yield pd.concat(out, ignore_index=True)

    # tail (optional)
    if buf_rows:
        yield pd.concat(buf, ignore_index=True)


"""

import pytest
import tempfile
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pelinker.io.parquet import ParquetWriter

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {
            "pmid": "pmid_0",
            "property": "property_0",
            "mention": "mention_0",
            "embed": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        {
            "pmid": "pmid_1",
            "property": "property_1",
            "mention": "mention_1",
            "embed": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        {
            "pmid": "pmid_2",
            "property": "property_2",
            "mention": "mention_2",
            "embed": [2.0, 3.0, 4.0, 5.0, 6.0],
        },
    ]


@pytest.fixture
def varied_embedding_data():
    """Create data with varied embedding dimensions."""
    return [
        {
            "pmid": "pmid_1",
            "property": "property_1",
            "mention": "mention_1",
            "embed": [1.0, 2.0, 3.0],  # 3D embedding
        },
        {
            "pmid": "pmid_2",
            "property": "property_2",
            "mention": "mention_2",
            "embed": [4.0, 5.0, 6.0, 7.0, 8.0],  # 5D embedding
        },
    ]


@pytest.fixture
def large_embedding_data():
    """Create data with large embeddings for performance testing."""
    data = []
    for i in range(100):
        # Create a 768-dimensional embedding (typical BERT size)
        embedding = [float(j + i * 0.1) for j in range(768)]
        data.append(
            {
                "pmid": f"pmid_{i}",
                "property": f"property_{i % 10}",  # 10 different properties
                "mention": f"mention_{i}",
                "embed": embedding,
            }
        )
    return data


# Helper functions
def read_parquet_file(file_path: pathlib.Path) -> pa.Table:
    """Helper to read parquet file and return as PyArrow table."""
    return pq.read_table(file_path)


# Tests
def test_basic_write_record_batch(temp_dir, sample_data):
    """Test basic writing functionality with RecordBatch."""
    output_path = temp_dir / "test_basic_rb.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(sample_data)
    finally:
        writer.close()

    # Verify file exists
    assert output_path.exists()

    # Verify content
    table = read_parquet_file(output_path)
    assert table.num_rows == 3
    assert table.num_columns == 4

    # Check column names
    expected_columns = ["pmid", "property", "mention", "embed"]
    assert table.column_names == expected_columns

    # Check data types
    assert table.schema.field("pmid").type == pa.string()
    assert table.schema.field("property").type == pa.string()
    assert table.schema.field("mention").type == pa.string()
    assert table.schema.field("embed").type == pa.list_(pa.float64())


def test_basic_write_table(temp_dir, sample_data):
    """Test basic writing functionality with Table."""
    output_path = temp_dir / "test_basic_table.parquet"

    writer = ParquetWriter(output_path, use_record_batch=False)
    try:
        writer.write_batch(sample_data)
    finally:
        writer.close()

    # Verify file exists and content
    assert output_path.exists()
    table = read_parquet_file(output_path)
    assert table.num_rows == 3
    assert table.num_columns == 4


def test_multiple_batches(temp_dir, sample_data):
    """Test writing multiple batches to the same file."""
    output_path = temp_dir / "test_multiple_batches.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        # Write first batch
        writer.write_batch(sample_data[:2])

        # Write second batch
        writer.write_batch(sample_data[2:])
    finally:
        writer.close()

    # Verify total rows
    table = read_parquet_file(output_path)
    assert table.num_rows == 3
    assert writer.total_rows == 3


def test_empty_batch_handling(temp_dir):
    """Test that empty batches are handled gracefully."""
    output_path = temp_dir / "test_empty_batch.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        # Write empty batch - should be skipped
        writer.write_batch([])

        # File should not exist yet
        assert not output_path.exists()

        # Write some actual data
        sample_data = [
            {
                "pmid": "pmid_1",
                "property": "property_1",
                "mention": "mention_1",
                "embed": [1.0, 2.0],
            }
        ]
        writer.write_batch(sample_data)

        # Now file should exist
        assert output_path.exists()

    finally:
        writer.close()

    table = read_parquet_file(output_path)
    assert table.num_rows == 1


def test_schema_consistency_across_batches(temp_dir):
    """Test that schema remains consistent across multiple batches."""
    output_path = temp_dir / "test_schema_consistency.parquet"

    batch1 = [
        {
            "pmid": "pmid_1",
            "property": "property_1",
            "mention": "mention_1",
            "embed": [1.0, 2.0, 3.0],
        }
    ]

    batch2 = [
        {
            "pmid": "pmid_2",
            "property": "property_2",
            "mention": "mention_2",
            "embed": [4.0, 5.0, 6.0, 7.0],  # Different embedding size
        }
    ]

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(batch1)
        writer.write_batch(batch2)
    finally:
        writer.close()

    # Verify both batches were written
    table = read_parquet_file(output_path)
    assert table.num_rows == 2

    # Verify schema consistency
    embed_column = table.column("embed")
    assert all(isinstance(row, list) for row in embed_column.to_pylist())


def test_large_embeddings_performance(temp_dir, large_embedding_data):
    """Test performance with large embeddings (768-dimensional)."""
    output_path = temp_dir / "test_large_embeddings.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(large_embedding_data)
    finally:
        writer.close()

    # Verify all data was written
    table = read_parquet_file(output_path)
    assert table.num_rows == 100

    # Verify embedding dimensions
    embed_column = table.column("embed").to_pylist()
    assert len(embed_column[0]) == 768
    assert all(len(embed) == 768 for embed in embed_column)


def test_data_integrity(temp_dir, sample_data):
    """Test that data is written and read back correctly."""
    output_path = temp_dir / "test_data_integrity.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(sample_data)
    finally:
        writer.close()

    # Read back and verify
    table = read_parquet_file(output_path)

    # Convert to list of dicts for comparison
    pmids = table.column("pmid").to_pylist()
    properties = table.column("property").to_pylist()
    mentions = table.column("mention").to_pylist()
    embeds = table.column("embed").to_pylist()

    for i, original_row in enumerate(sample_data):
        assert pmids[i] == original_row["pmid"]
        assert properties[i] == original_row["property"]
        assert mentions[i] == original_row["mention"]
        assert embeds[i] == original_row["embed"]


def test_error_handling_invalid_data(temp_dir):
    """Test error handling with invalid data types."""
    output_path = temp_dir / "test_error_handling.parquet"

    # Invalid data - pmid should be string, not int
    invalid_data = [
        {
            "pmid": 123,  # Should be string
            "property": "property_1",
            "mention": "mention_1",
            "embed": [1.0, 2.0],
        }
    ]

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        with pytest.raises(Exception):  # Should raise some kind of type error
            writer.write_batch(invalid_data)
    finally:
        writer.close()


def test_context_manager_usage(temp_dir, sample_data):
    """Test using ParquetWriter as a context manager (if implemented)."""
    output_path = temp_dir / "test_context_manager.parquet"

    # For now, test the manual close pattern
    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(sample_data)
        assert writer.total_rows == 3
    finally:
        writer.close()

    # Verify file was properly closed and written
    assert output_path.exists()
    table = read_parquet_file(output_path)
    assert table.num_rows == 3


def test_different_embedding_sizes_same_batch(temp_dir, varied_embedding_data):
    """Test handling of different embedding sizes in the same batch."""
    output_path = temp_dir / "test_varied_embeddings.parquet"

    writer = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer.write_batch(varied_embedding_data)
    finally:
        writer.close()

    # Verify both rows were written despite different embedding sizes
    table = read_parquet_file(output_path)
    assert table.num_rows == 2

    # Check that embeddings have different lengths
    embed_column = table.column("embed").to_pylist()
    assert len(embed_column[0]) == 3
    assert len(embed_column[1]) == 5


def test_file_overwrite_behavior(temp_dir, sample_data):
    """Test behavior when writing to an existing file."""
    output_path = temp_dir / "test_overwrite.parquet"

    # Write first file
    writer1 = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer1.write_batch(sample_data[:1])
    finally:
        writer1.close()

    # Verify first file
    table1 = read_parquet_file(output_path)
    assert table1.num_rows == 1

    # Write second file (should overwrite)
    writer2 = ParquetWriter(output_path, use_record_batch=True)
    try:
        writer2.write_batch(sample_data)
    finally:
        writer2.close()

    # Verify second file overwrote the first
    table2 = read_parquet_file(output_path)
    assert table2.num_rows == 3


if __name__ == "__main__":
    pytest.main([__file__])
