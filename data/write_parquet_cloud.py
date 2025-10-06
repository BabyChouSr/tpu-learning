"""
Example showing how streaming ParquetWriter.write_batch() works with different storage backends.

The same pattern works for:
- Local filesystem
- AWS S3 (s3://)
- Google Cloud Storage (gs://)
- Azure Blob Storage (az://)
- HDFS (hdfs://)
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.fs as pafs


def map_increment_days(batch, increment=1):
    """Map operation: increment the days column by a given value."""
    days_array = batch['days']
    incremented_days = pa.compute.add(days_array, increment)
    incremented_days = pa.compute.cast(incremented_days, days_array.type)
    
    new_batch = pa.record_batch(
        [incremented_days, batch['months'], batch['years']],
        names=['days', 'months', 'years']
    )
    return new_batch


def stream_map_batches(input_path, transform_fn, batch_size=10, filesystem=None, **transform_kwargs):
    """
    Generator that reads batches from parquet and applies a transformation.
    Works with any PyArrow filesystem.
    """
    dataset = ds.dataset(input_path, format="parquet", filesystem=filesystem)
    batch_num = 0
    
    for batch in dataset.to_batches(batch_size=batch_size):
        print(f"Reading batch {batch_num}: {len(batch)} rows")
        transformed_batch = transform_fn(batch, **transform_kwargs)
        batch_num += 1
        yield transformed_batch


def streaming_write_parquet(input_path, output_path, transform_fn, 
                            batch_size=10, input_fs=None, output_fs=None, 
                            **transform_kwargs):
    """
    Stream read -> transform -> write parquet with support for different filesystems.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        transform_fn: Function to transform each batch
        batch_size: Number of rows per batch
        input_fs: PyArrow filesystem for input (None = local)
        output_fs: PyArrow filesystem for output (None = local)
        **transform_kwargs: Additional kwargs for transform function
    """
    # Get schema from input
    dataset = ds.dataset(input_path, format="parquet", filesystem=input_fs)
    schema = dataset.schema
    
    print(f"Schema: {schema}")
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    
    # Open output file with specified filesystem
    if output_fs is None:
        # Local filesystem - can use simple path
        writer = pq.ParquetWriter(output_path, schema)
    else:
        # Cloud filesystem - need to open file handle
        with output_fs.open_output_stream(output_path) as out_stream:
            writer = pq.ParquetWriter(out_stream, schema)
            try:
                for batch in stream_map_batches(input_path, transform_fn, 
                                               batch_size=batch_size, 
                                               filesystem=input_fs,
                                               **transform_kwargs):
                    writer.write_batch(batch)
            finally:
                writer.close()
        return
    
    # Local filesystem path
    try:
        for batch in stream_map_batches(input_path, transform_fn, 
                                       batch_size=batch_size, 
                                       filesystem=input_fs,
                                       **transform_kwargs):
            writer.write_batch(batch)
    finally:
        writer.close()


# ============================================================================
# Example 1: Local filesystem (current approach)
# ============================================================================
print("="*70)
print("Example 1: Local Filesystem")
print("="*70)

input_path = 'data/samples/test.parquet'
output_path = 'data/samples/streaming_local.parquet'

streaming_write_parquet(
    input_path=input_path,
    output_path=output_path,
    transform_fn=map_increment_days,
    batch_size=10,
    increment=1
)

# Verify
table = pq.read_table(output_path)
print(f"✓ Wrote {len(table)} rows to local filesystem")
print(f"  Days (first 5): {table['days'][:5].to_pylist()}\n")


# ============================================================================
# Example 2: Google Cloud Storage (GCS)
# ============================================================================
print("="*70)
print("Example 2: Google Cloud Storage (requires credentials)")
print("="*70)

try:
    from google.cloud import storage
    from pyarrow.fs import GcsFileSystem
    
    # Initialize GCS filesystem
    # Note: Requires GOOGLE_APPLICATION_CREDENTIALS env var or default credentials
    gcs = GcsFileSystem(anonymous=False)
    
    # Example paths (replace with your bucket)
    # gcs_input = "your-bucket/data/input.parquet"
    # gcs_output = "your-bucket/data/output.parquet"
    
    print("GCS filesystem available!")
    print("Usage:")
    print("  streaming_write_parquet(")
    print("      input_path='your-bucket/data/input.parquet',")
    print("      output_path='your-bucket/data/output.parquet',")
    print("      transform_fn=map_increment_days,")
    print("      input_fs=gcs,")
    print("      output_fs=gcs,")
    print("      increment=1")
    print("  )")
    
except ImportError:
    print("GCS support not installed. Install with:")
    print("  pip install google-cloud-storage")
    print("\nThe pattern would be:")
    print("  from pyarrow.fs import GcsFileSystem")
    print("  gcs = GcsFileSystem()")
    print("  streaming_write_parquet(..., input_fs=gcs, output_fs=gcs)")

print()


# ============================================================================
# Example 3: AWS S3
# ============================================================================
print("="*70)
print("Example 3: AWS S3 (requires credentials)")
print("="*70)

try:
    from pyarrow.fs import S3FileSystem
    
    # Initialize S3 filesystem
    # Uses AWS credentials from environment or ~/.aws/credentials
    s3 = S3FileSystem(
        region='us-east-1',  # or your region
        # access_key='YOUR_ACCESS_KEY',  # optional
        # secret_key='YOUR_SECRET_KEY',  # optional
    )
    
    print("S3 filesystem available!")
    print("Usage:")
    print("  streaming_write_parquet(")
    print("      input_path='my-bucket/data/input.parquet',")
    print("      output_path='my-bucket/data/output.parquet',")
    print("      transform_fn=map_increment_days,")
    print("      input_fs=s3,")
    print("      output_fs=s3,")
    print("      increment=1")
    print("  )")
    
except Exception as e:
    print(f"S3 support issue: {e}")
    print("\nThe pattern would be:")
    print("  from pyarrow.fs import S3FileSystem")
    print("  s3 = S3FileSystem(region='us-east-1')")
    print("  streaming_write_parquet(..., input_fs=s3, output_fs=s3)")

print()


# ============================================================================
# Example 4: Azure Blob Storage
# ============================================================================
print("="*70)
print("Example 4: Azure Blob Storage (requires credentials)")
print("="*70)

try:
    # Azure support requires additional setup
    print("Azure Blob Storage pattern:")
    print("  # Note: PyArrow doesn't have direct Azure support")
    print("  # Option 1: Use fsspec with adlfs")
    print("  from pyarrow.fs import PyFileSystem, FSSpecHandler")
    print("  import adlfs")
    print("  ")
    print("  fs = adlfs.AzureBlobFileSystem(account_name='...', account_key='...')")
    print("  pa_fs = PyFileSystem(FSSpecHandler(fs))")
    print("  streaming_write_parquet(..., input_fs=pa_fs, output_fs=pa_fs)")
    print()
    print("  # Option 2: Use abfs:// URLs with fsspec backend")
    
except Exception as e:
    print(f"Azure info: {e}")

print()


# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("SUMMARY: How it works")
print("="*70)
print("""
The streaming write pattern works universally because:

1. PyArrow abstracts filesystems via `pyarrow.fs`
2. ParquetWriter can write to any file-like object
3. Datasets can read from any filesystem

Key points:
✓ Local: Just use file paths (default)
✓ Cloud: Pass filesystem object via `filesystem` parameter
✓ Same code pattern for all backends
✓ Streaming works identically - only one batch in memory at a time
✓ No need to download entire files - operations happen remotely

Performance considerations:
- Network latency affects batch I/O
- Use larger batch sizes for cloud storage (100-1000 rows)
- Cloud providers often have better PyArrow integration in their native SDKs
- For GCS: Consider using google-cloud-storage directly
- For S3: S3FileSystem is well-optimized
""")

