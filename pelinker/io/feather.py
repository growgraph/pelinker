from pelinker.io.reader import read_batches


def read_feather_mmap_batches(file_path, batch_size=1000):
    """
    Read feather file using memory mapping for large files.

    This is a convenience wrapper around the unified read_batches function.
    For new code, prefer using pelinker.io.read_batches directly.
    """
    yield from read_batches(file_path, batch_size=batch_size, file_type="feather")
