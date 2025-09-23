import pathlib
from typing import Optional, List

import pyarrow as pa
from pyarrow import parquet as pq
import logging

logger = logging.getLogger(__name__)


class ParquetWriter:
    """
    Wrapper class for safer Parquet writing with direct PyArrow RecordBatch construction
    """

    def __init__(self, output_path: pathlib.Path, use_record_batch: bool = True):
        self.output_path = output_path
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.total_rows = 0
        self.use_record_batch = use_record_batch

    def _get_schema(self) -> pa.Schema:
        """Define schema explicitly - no guessing from data"""
        fields = [
            pa.field("pmid", pa.string()),
            pa.field("property", pa.string()),
            pa.field("mention", pa.string()),
            # For embeddings, we'll use list of float64
            pa.field("embed", pa.list_(pa.float64())),
        ]
        return pa.schema(fields)

    def _dict_list_to_arrow_arrays(self, data: List[dict]) -> tuple:
        """Convert list of dicts directly to PyArrow arrays"""
        if not data:
            raise ValueError("Cannot create arrays from empty data")

        # Extract columns
        pmids = [row["pmid"] for row in data]
        properties = [row["property"] for row in data]
        mentions = [row["mention"] for row in data]
        embeds = [row["embed"] for row in data]

        # Create PyArrow arrays directly with explicit types
        pmid_array = pa.array(pmids, type=pa.string())
        property_array = pa.array(properties, type=pa.string())
        mention_array = pa.array(mentions, type=pa.string())
        embed_array = pa.array(embeds, type=pa.list_(pa.float64()))

        return pmid_array, property_array, mention_array, embed_array

    def _dict_list_to_arrow_table(self, data: List[dict]) -> pa.Table:
        """Convert list of dicts directly to PyArrow table without pandas"""
        arrays = self._dict_list_to_arrow_arrays(data)
        return pa.table(arrays, schema=self.schema)

    def _dict_list_to_record_batch(self, data: List[dict]) -> pa.RecordBatch:
        """Convert list of dicts directly to PyArrow RecordBatch (more memory efficient)"""
        arrays = self._dict_list_to_arrow_arrays(data)
        return pa.record_batch(list(arrays), schema=self.schema)

    def write_batch(self, data: List[dict]):
        """Write a batch of data to Parquet"""
        if not data:
            logger.info("Skipping empty batch")
            return

        try:
            # Initialize schema and writer on first batch
            if self.writer is None:
                self.schema = self._get_schema()
                self.writer = pq.ParquetWriter(self.output_path, self.schema)
                logger.info(f"Initialized Parquet writer with schema: {self.schema}")

            if self.use_record_batch:
                # Use RecordBatch for better memory efficiency
                batch = self._dict_list_to_record_batch(data)
                self.writer.write_batch(batch)
            else:
                # Use Table (slightly more memory but might be more compatible)
                table = self._dict_list_to_arrow_table(data)
                self.writer.write_table(table)

            self.total_rows += len(data)
            logger.info(f"Wrote batch with {len(data)} rows. Total: {self.total_rows}")

        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            raise

    def close(self):
        """Close the writer"""
        if self.writer is not None:
            self.writer.close()
            logger.info(f"Closed Parquet writer. Total rows written: {self.total_rows}")
