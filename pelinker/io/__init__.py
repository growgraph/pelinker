"""
Unified IO interface for reading large files in chunks.
Supports Feather, Parquet, and CSV/TSV formats.
"""

from pelinker.io.reader import read_batches

__all__ = ["read_batches"]
