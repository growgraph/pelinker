"""
Unified reader interface for reading large files in chunks.
Supports Feather, Parquet, and CSV/TSV formats.
"""

import pathlib
from typing import Iterator, Optional

import pandas as pd
from pyarrow import feather as pf
from pyarrow import parquet as pq


def _detect_file_type(file_path: str) -> str:
    """Detect file type from extension, handling compressed files."""
    path = pathlib.Path(file_path)
    suffixes = [s.lower() for s in path.suffixes]

    # Handle compressed files (e.g., .csv.gz, .tsv.gz)
    # Remove compression extensions (.gz, .bz2, .xz, .zip)
    compression_exts = {".gz", ".bz2", ".xz", ".zip"}
    base_suffixes = [s for s in suffixes if s not in compression_exts]

    if not base_suffixes:
        raise ValueError(
            f"Could not detect file type from path: {file_path}. "
            "Supported formats: .feather, .parquet, .csv, .tsv (optionally compressed)"
        )

    suffix = base_suffixes[-1]  # Get the last suffix (before compression)

    if suffix == ".feather":
        return "feather"
    elif suffix == ".parquet":
        return "parquet"
    elif suffix in (".csv", ".tsv"):
        return "csv"
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Supported formats: .feather, .parquet, .csv, .tsv (optionally compressed)"
        )


def _read_feather_batches(file_path: str, batch_size: int) -> Iterator[pd.DataFrame]:
    """Read feather file in batches using memory mapping."""
    table = pf.read_table(file_path)
    total_rows = len(table)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_table = table.slice(start_idx, end_idx - start_idx)
        yield batch_table.to_pandas()


def _read_parquet_batches(
    file_path: str, batch_size: int, columns: Optional[list] = None
) -> Iterator[pd.DataFrame]:
    """Read parquet file in batches."""
    parquet_file = pq.ParquetFile(file_path)

    for batch_group in parquet_file.iter_batches(
        batch_size=batch_size, columns=columns
    ):
        yield batch_group.to_pandas()


def _read_csv_batches(
    file_path: str, batch_size: int, sep: Optional[str] = None, **kwargs
) -> Iterator[pd.DataFrame]:
    """Read CSV/TSV file in batches, including compressed files."""
    # Auto-detect separator if not provided
    if sep is None:
        path = pathlib.Path(file_path)
        # Check base suffix (before compression extension)
        suffixes = [s.lower() for s in path.suffixes]
        compression_exts = {".gz", ".bz2", ".xz", ".zip"}
        base_suffixes = [s for s in suffixes if s not in compression_exts]
        base_suffix = base_suffixes[-1] if base_suffixes else ".csv"
        sep = "\t" if base_suffix == ".tsv" else ","

    # Use pandas read_csv with chunksize (handles compression automatically)
    for chunk in pd.read_csv(file_path, chunksize=batch_size, sep=sep, **kwargs):
        yield chunk


def read_batches(
    file_path: str, batch_size: int = 1000, file_type: Optional[str] = None, **kwargs
) -> Iterator[pd.DataFrame]:
    """
    Read large files in batches, supporting Feather, Parquet, and CSV/TSV formats.

    Automatically detects file type from extension if not provided.

    Args:
        file_path: Path to the file to read
        batch_size: Number of rows per batch (default: 1000)
        file_type: Optional file type override ('feather', 'parquet', 'csv').
                   If None, auto-detects from file extension.
        **kwargs: Additional arguments passed to format-specific readers:
                  - For CSV: sep, header, etc. (pandas.read_csv arguments)
                  - For Parquet: columns (list of column names to read)

    Yields:
        pd.DataFrame: Batches of data as pandas DataFrames

    Examples:
        >>> # Read feather file
        >>> for batch in read_batches("data.feather", batch_size=5000):
        ...     process(batch)

        >>> # Read parquet file
        >>> for batch in read_batches("data.parquet", batch_size=10000):
        ...     process(batch)

        >>> # Read CSV file with custom separator
        >>> for batch in read_batches("data.csv", batch_size=2000, sep=";"):
        ...     process(batch)
    """
    if file_type is None:
        file_type = _detect_file_type(file_path)

    if file_type == "feather":
        yield from _read_feather_batches(file_path, batch_size)
    elif file_type == "parquet":
        # Extract columns from kwargs if provided
        columns = kwargs.pop("columns", None)
        yield from _read_parquet_batches(file_path, batch_size, columns=columns)
    elif file_type == "csv":
        yield from _read_csv_batches(file_path, batch_size, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
