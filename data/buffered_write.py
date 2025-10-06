import os
from typing import Iterator, Callable, Optional
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

class BufferedParquetWriter:
    """
    Buffers batches in memory and writes to disk when:
    1. Buffer reaches a size threshold (write batches efficiently)
    2. File reaches a size threshold (start new file part)
    """
    
    def __init__(
        self,
        output_dir: str,
        schema: pa.Schema,
        batch_buffer_size: int = 100,  # Number of batches to buffer before writing
        file_size_mb: int = 128,  # Max file size in MB before starting new file
        prefix: str = "part"
    ):
        self.output_dir = output_dir
        self.schema = schema
        self.batch_buffer_size = batch_buffer_size
        self.file_size_bytes = file_size_mb * 1024 * 1024
        self.prefix = prefix
        
        os.makedirs(output_dir, exist_ok=True)
        
        # State
        self.current_batches = []
        self.current_file_num = 0
        self.current_file_size = 0
        self.total_rows_written = 0
        
    def _get_file_path(self, file_num: int) -> str:
        """Get the path for a numbered part file."""
        return os.path.join(self.output_dir, f"{self.prefix}-{file_num:05d}.parquet")
    
    def _write_buffered_batches(self):
        """Write accumulated batches to current or new file."""
        if not self.current_batches:
            return
        
        # Combine batches into a table
        table = pa.Table.from_batches(self.current_batches, schema=self.schema)
        
        # Estimate size (rough approximation - actual size may vary with compression)
        estimated_size = table.nbytes
        
        # Check if we need a new file
        if self.current_file_size > 0 and (self.current_file_size + estimated_size) > self.file_size_bytes:
            print(f"  File {self.current_file_num} reached size threshold, starting new file")
            self.current_file_num += 1
            self.current_file_size = 0
        
        # Write the table
        file_path = self._get_file_path(self.current_file_num)
        
        if self.current_file_size == 0:
            # New file - overwrite
            pq.write_table(table, file_path)
            print(f"  Created new file: {file_path} with {len(table)} rows")
        else:
            # Append to existing file by reading + writing
            start = time.time()
            existing_table = pq.read_table(file_path)
            combined_table = pa.concat_tables([existing_table, table])
            end = time.time()
            print(f"  Concatenated in {end - start} seconds")

            start = time.time()
            pq.write_table(combined_table, file_path)
            end = time.time()
            print(f"  Appended in {end - start} seconds")
            print(f"  Appended to {file_path}, now has {len(combined_table)} rows")
        
        # Update state
        self.current_file_size += estimated_size
        self.total_rows_written += len(table)
        self.current_batches = []
        
        # Get actual file size
        actual_size = os.path.getsize(file_path)
        print(f"    Actual file size: {actual_size / (1024*1024):.2f} MB")
    
    def write_batch(self, batch: pa.RecordBatch):
        """Add a batch to the buffer and flush if needed."""
        self.current_batches.append(batch)
        
        # Check if buffer is full
        if len(self.current_batches) >= self.batch_buffer_size:
            print(f"Buffer full ({len(self.current_batches)} batches), flushing to disk...")
            self._write_buffered_batches()
    
    def close(self):
        """Flush any remaining batches and close."""
        if self.current_batches:
            print(f"Flushing remaining {len(self.current_batches)} batches...")
            self._write_buffered_batches()
        
        print(f"\nWriting complete:")
        print(f"  Total rows written: {self.total_rows_written}")
        print(f"  Total files created: {self.current_file_num + 1}")
        print(f"  Output directory: {self.output_dir}")


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


def stream_map_batches(input_path, transform_fn, batch_size=10, **transform_kwargs):
    """Generator that reads batches from parquet and applies a transformation."""
    input_dataset = ds.dataset(input_path, format="parquet")
    batch_num = 0
    
    for batch in input_dataset.to_batches(batch_size=batch_size):
        print(f"Reading batch {batch_num}: days = {batch['days'].to_pylist()}")
        
        # Apply the transformation
        transformed_batch = transform_fn(batch, **transform_kwargs)
        
        print(f"  -> Transformed: days = {transformed_batch['days'].to_pylist()}")
        batch_num += 1
        yield transformed_batch


# Example usage
if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    days = pa.array([1, 2, 3] * 100, type=pa.int16())  # Larger dataset
    months = pa.array([4, 5, 6] * 100, type=pa.int16())
    years = pa.array([2020, 2021, 2022] * 100, type=pa.int16())
    table = pa.table([days, months, years], names=['days', 'months', 'years'])
    
    os.makedirs('data/samples', exist_ok=True)
    input_path = 'data/samples/test_large.parquet'
    pq.write_table(table, input_path)
    print(f"Created input file with {len(table)} rows\n")
    
    # Use buffered writer
    output_dir = 'data/samples/buffered_output'
    schema = table.schema
    
    writer = BufferedParquetWriter(
        output_dir=output_dir,
        schema=schema,
        batch_buffer_size=10,  # Write to disk every 10 batches
        file_size_mb=1,  # Start new file after 1MB (small for demo)
        prefix="part"
    )
    
    try:
        for batch in stream_map_batches(input_path, map_increment_days, batch_size=10, increment=1):
            writer.write_batch(batch)
    finally:
        writer.close()
    
    # Verify by reading back as a dataset
    print("\n" + "="*50)
    print("Reading back all files as a single dataset:")
    print("="*50)
    
    output_dataset = ds.dataset(output_dir, format="parquet")
    result_table = output_dataset.to_table()
    
    print(f"Total rows read: {len(result_table)}")
    print(f"Days (first 10): {result_table['days'][:10].to_pylist()}")
    print(f"Days (last 10): {result_table['days'][-10:].to_pylist()}")
    
    # Show all files created
    print(f"\nFiles created in {output_dir}:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.parquet'):
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_table = pq.read_table(file_path)
            print(f"  {file}: {len(file_table)} rows, {size_mb:.2f} MB")

