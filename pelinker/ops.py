import pathlib
from typing import Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _detect_file_format(file_path: pathlib.Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in [".tsv", ".tsv.gz"]:
        return "tsv"
    elif suffix in [".csv", ".csv.gz"]:
        return "csv"
    else:
        # Default to tsv for backward compatibility
        logger.warning(f"Unknown file extension {suffix}, defaulting to TSV format")
        return "tsv"


def _detect_headers_and_columns(
    file_path: pathlib.Path, file_format: str
) -> Tuple[bool, str, str]:
    """
    Detect if file has headers and determine column names.
    Returns: (has_header, pmid_col, text_col)
    """
    # Read first few lines to detect headers
    try:
        if file_format == "tsv":
            sample_df = pd.read_csv(
                file_path,
                sep="\t",
                nrows=5,
                compression="gzip" if file_path.suffix.endswith(".gz") else None,
            )
        else:  # csv
            sample_df = pd.read_csv(
                file_path,
                sep=",",
                nrows=5,
                compression="gzip" if file_path.suffix.endswith(".gz") else None,
            )
    except Exception as e:
        logger.warning(f"Could not read sample for header detection: {e}")
        return False, 0, 1

    # Check if first row looks like headers (contains text-like values)
    first_row = sample_df.iloc[0]

    # Look for common header patterns
    pmid_candidates = ["pmid", "PMID", "id", "ID", "paper_id", "document_id"]
    text_candidates = ["abstract", "text", "content", "body", "description"]

    pmid_col = None
    text_col = None

    # Check column names for pmid and text
    for col in sample_df.columns:
        col_lower = str(col).lower()
        if any(candidate in col_lower for candidate in pmid_candidates):
            pmid_col = col
        if any(candidate in col_lower for candidate in text_candidates):
            text_col = col

    # If we found both expected columns, assume headers exist
    if pmid_col is not None and text_col is not None:
        logger.info(f"Detected headers: pmid_col='{pmid_col}', text_col='{text_col}'")
        return True, pmid_col, text_col

    # Check if first row contains numeric values (likely data, not headers)
    has_numeric_first_row = any(
        pd.api.types.is_numeric_dtype(sample_df[col])
        or str(first_row[col]).replace(".", "").replace("-", "").isdigit()
        for col in sample_df.columns
    )

    if has_numeric_first_row:
        logger.info("No headers detected, using column indices 0 and 1")
        return False, 0, 1
    else:
        # First row might be headers, use column names
        logger.info("Headers detected, using column names")
        return True, sample_df.columns[0], sample_df.columns[1]
