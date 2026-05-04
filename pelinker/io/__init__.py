"""
Unified IO interface for reading large files in chunks.
Supports Feather, Parquet, and CSV/TSV formats.
"""

from pelinker.io.json_files import is_gzip_file_path, load_json_path
from pelinker.io.reader import read_batches

__all__ = ["is_gzip_file_path", "load_json_path", "read_batches"]
