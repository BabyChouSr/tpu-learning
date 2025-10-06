import os
import json
import gzip
import zstandard as zstd

import pyarrow as pa
import pyarrow.parquet as pq

# Create sample data
days = pa.array([1, 2, 3] * 10, type=pa.int16())
months = pa.array([4, 5, 6] * 10, type=pa.int16())
years = pa.array([2020, 2021, 2022] * 10, type=pa.int16())
table = pa.table([days, months, years], names=['days', 'months', 'years'])

os.makedirs('data/samples', exist_ok=True)

# Write to gzip-compressed JSONL (write batches directly)
print("Writing to gzip-compressed JSONL using PyArrow batches...")
with gzip.open('data/samples/test.jsonl.gz', 'wt', encoding='utf-8') as f:
    for batch in table.to_batches(max_chunksize=10):
        # Convert batch to list of dicts and write as JSONL
        for record in batch.to_pylist():
            json.dump(record, f)
            f.write('\n')

# Write to zstandard-compressed JSONL (write batches directly)
print("Writing to zstandard-compressed JSONL using PyArrow batches...")
cctx = zstd.ZstdCompressor()
with open('data/samples/test.jsonl.zst', 'wb') as f:
    with cctx.stream_writer(f) as compressor:
        for batch in table.to_batches(max_chunksize=10):
            # Convert batch to list of dicts and write as JSONL
            for record in batch.to_pylist():
                line = json.dumps(record) + '\n'
                compressor.write(line.encode('utf-8'))

print("\n" + "="*50)
print("Reading back compressed JSONL files as PyArrow batches")
print("="*50)

def read_jsonl_gz_batches(filepath, batch_size=10):
    """Generator that reads gzip-compressed JSONL as PyArrow RecordBatches."""
    batch_records = []
    batch_num = 0
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            batch_records.append(json.loads(line))
            
            if len(batch_records) >= batch_size:
                # Convert to PyArrow RecordBatch
                batch = pa.RecordBatch.from_pylist(batch_records)
                print(f"Batch {batch_num}: {batch}")
                yield batch
                batch_num += 1
                batch_records = []
        
        # Yield remaining items
        if batch_records:
            batch = pa.RecordBatch.from_pylist(batch_records)
            print(f"Batch {batch_num}: {batch}")
            yield batch

def read_jsonl_zst_batches(filepath, batch_size=10):
    """Generator that reads zstandard-compressed JSONL as PyArrow RecordBatches."""
    batch_records = []
    batch_num = 0
    
    dctx = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line:  # Skip empty lines
                    batch_records.append(json.loads(line))
                    
                    if len(batch_records) >= batch_size:
                        # Convert to PyArrow RecordBatch
                        batch = pa.RecordBatch.from_pylist(batch_records)
                        print(f"Batch {batch_num}: {batch}")
                        yield batch
                        batch_num += 1
                        batch_records = []
            
            # Yield remaining items
            if batch_records:
                batch = pa.RecordBatch.from_pylist(batch_records)
                print(f"Batch {batch_num}: {batch}")
                yield batch

print("\nReading from .jsonl.gz as PyArrow batches:")
for batch in read_jsonl_gz_batches('data/samples/test.jsonl.gz', batch_size=10):
    # Access columns like PyArrow
    print(f"  Days column: {batch['days'].to_pylist()}")

print("\nReading from .jsonl.zst as PyArrow batches:")
for batch in read_jsonl_zst_batches('data/samples/test.jsonl.zst', batch_size=10):
    # Access columns like PyArrow
    print(f"  Days column: {batch['days'].to_pylist()}")

print("\n" + "="*50)
print("Testing streaming write with MAP operation using PyArrow batches")
print("="*50)

def map_increment_days(batch, increment=1):
    """Map operation: increment the days column by a given value (PyArrow batch)."""
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

def stream_map_jsonl_gz(input_path, output_path, transform_fn, batch_size=10, **transform_kwargs):
    """Stream read from gzip JSONL as PyArrow batches, apply transformation, and write."""
    batch_num = 0
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as fout:
        for batch in read_jsonl_gz_batches(input_path, batch_size):
            print(f"Processing batch {batch_num}: days = {batch['days'].to_pylist()}")
            
            # Apply transformation to batch
            transformed_batch = transform_fn(batch, **transform_kwargs)
            
            print(f"  -> Transformed: days = {transformed_batch['days'].to_pylist()}")
            
            # Write transformed batch as JSONL
            for record in transformed_batch.to_pylist():
                json.dump(record, fout)
                fout.write('\n')
            
            batch_num += 1

def stream_map_jsonl_zst(input_path, output_path, transform_fn, batch_size=10, **transform_kwargs):
    """Stream read from zstandard JSONL as PyArrow batches, apply transformation, and write."""
    batch_num = 0
    
    cctx = zstd.ZstdCompressor()
    with open(output_path, 'wb') as fout:
        with cctx.stream_writer(fout) as writer:
            for batch in read_jsonl_zst_batches(input_path, batch_size):
                print(f"Processing batch {batch_num}: days = {batch['days'].to_pylist()}")
                
                # Apply transformation to batch
                transformed_batch = transform_fn(batch, **transform_kwargs)
                
                print(f"  -> Transformed: days = {transformed_batch['days'].to_pylist()}")
                
                # Write transformed batch as JSONL
                for record in transformed_batch.to_pylist():
                    line = json.dumps(record) + '\n'
                    writer.write(line.encode('utf-8'))
                
                batch_num += 1

# Stream write with map operation for gzip
print("\n--- Gzip streaming with map operation ---")
input_gz = 'data/samples/test.jsonl.gz'
output_gz = 'data/samples/streaming_mapped.jsonl.gz'

# Suppress the batch printing for cleaner output
def read_jsonl_gz_batches_quiet(filepath, batch_size=10):
    """Silent version for processing."""
    batch_records = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            batch_records.append(json.loads(line))
            if len(batch_records) >= batch_size:
                yield pa.RecordBatch.from_pylist(batch_records)
                batch_records = []
        if batch_records:
            yield pa.RecordBatch.from_pylist(batch_records)

# Use the quiet version for the final run
batch_num = 0
with gzip.open(output_gz, 'wt', encoding='utf-8') as fout:
    for batch in read_jsonl_gz_batches_quiet(input_gz, batch_size=10):
        print(f"Processing batch {batch_num}: days = {batch['days'].to_pylist()}")
        transformed_batch = map_increment_days(batch, increment=1)
        print(f"  -> Transformed: days = {transformed_batch['days'].to_pylist()}")
        
        for record in transformed_batch.to_pylist():
            json.dump(record, fout)
            fout.write('\n')
        batch_num += 1

print(f"\nSuccessfully wrote streaming gzip JSONL with map to {output_gz}")

# Read back to verify gzip (using PyArrow)
print("\nReading back the mapped streaming gzip JSONL file:")
all_batches = list(read_jsonl_gz_batches_quiet(output_gz, batch_size=1000))
result_table = pa.Table.from_batches(all_batches)

print(f"Total records: {len(result_table)}")
print(f"Days (first 10): {result_table['days'][:10].to_pylist()}")
print(f"Days (last 10): {result_table['days'][-10:].to_pylist()}")

# Stream write with map operation for zstandard
print("\n--- Zstandard streaming with map operation ---")
input_zst = 'data/samples/test.jsonl.zst'
output_zst = 'data/samples/streaming_mapped.jsonl.zst'

def read_jsonl_zst_batches_quiet(filepath, batch_size=10):
    """Silent version for processing."""
    batch_records = []
    dctx = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line:
                    batch_records.append(json.loads(line))
                    if len(batch_records) >= batch_size:
                        yield pa.RecordBatch.from_pylist(batch_records)
                        batch_records = []
            if batch_records:
                yield pa.RecordBatch.from_pylist(batch_records)

batch_num = 0
cctx = zstd.ZstdCompressor()
with open(output_zst, 'wb') as fout:
    with cctx.stream_writer(fout) as writer:
        for batch in read_jsonl_zst_batches_quiet(input_zst, batch_size=10):
            print(f"Processing batch {batch_num}: days = {batch['days'].to_pylist()}")
            transformed_batch = map_increment_days(batch, increment=1)
            print(f"  -> Transformed: days = {transformed_batch['days'].to_pylist()}")
            
            for record in transformed_batch.to_pylist():
                line = json.dumps(record) + '\n'
                writer.write(line.encode('utf-8'))
            batch_num += 1

print(f"\nSuccessfully wrote streaming zstandard JSONL with map to {output_zst}")

# Read back to verify zstandard (using PyArrow)
print("\nReading back the mapped streaming zstandard JSONL file:")
all_batches = list(read_jsonl_zst_batches_quiet(output_zst, batch_size=1000))
result_table = pa.Table.from_batches(all_batches)

print(f"Total records: {len(result_table)}")
print(f"Days (first 10): {result_table['days'][:10].to_pylist()}")
print(f"Days (last 10): {result_table['days'][-10:].to_pylist()}")

