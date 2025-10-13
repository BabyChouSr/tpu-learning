import os

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

days = pa.array([1, 2, 3] * 10, type=pa.int16())
months = pa.array([4, 5, 6] * 10, type=pa.int16())
years = pa.array([2020, 2021, 2022] * 10, type=pa.int16())
table = pa.table([days, months, years], names=['days', 'months', 'years'])

batch_num = 0
for batch in table.to_batches(10):
    print(f"Batch number: {batch_num}, batch: {batch}")

os.makedirs('data/samples', exist_ok=True)
pq.write_table(table, 'data/samples/test.parquet')

dataset = ds.dataset('data/samples/test.parquet', format="parquet")
for batch in dataset.to_batches(columns=['days', 'months', 'years'], batch_size=10):
    print(f"Batch number: {batch_num}, batch: {batch}")
    batch_num += 1

# os.makedirs('data/samples/dir-partitioning', exist_ok=True)
# ds.write_dataset(table, "data/samples/dir-partitioning", format="parquet",
#                  partitioning=ds.partitioning(
#                     pa.schema([table.schema.field("years")])
#                 ))

print("\n" + "="*50)
print("Testing streaming write with write_batch() + MAP operation")
print("="*50)

def map_increment_days(batch, increment=1):
    """Map operation: increment the days column by a given value."""
    # Get the days column and add increment
    days_array = batch['days']
    incremented_days = pa.compute.add(days_array, increment)
    
    # Cast back to original type to preserve schema
    incremented_days = pa.compute.cast(incremented_days, days_array.type)
    
    # Create new batch with incremented days
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

# Read from existing parquet and stream write with transformation
input_path = 'data/samples/test.parquet'
output_path = 'data/samples/streaming_mapped.parquet'

# Get schema from the input file
input_table = pq.read_table(input_path)
schema = input_table.schema

print(f"Input file has {len(input_table)} rows")
print(f"Original days (first 10): {input_table['days'][:10].to_pylist()}")

# Stream write with map operation
writer = pq.ParquetWriter(output_path, schema)

try:
    for batch in stream_map_batches(input_path, map_increment_days, batch_size=10, increment=1):
        writer.write_batch(batch)
finally:
    writer.close()

print(f"\nSuccessfully wrote streaming parquet with map to {output_path}")

# Read back to verify
print("\nReading back the mapped streaming parquet file:")
table_read = pq.read_table(output_path)
print(f"Total rows: {len(table_read)}")
print(f"Days column (first 10): {table_read['days'][:10].to_pylist()}")
print(f"Days column (last 10): {table_read['days'][-10:].to_pylist()}")