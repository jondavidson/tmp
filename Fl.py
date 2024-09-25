import pyarrow as pa
import pyarrow.flight

# Enable IPC compression (ZSTD or LZ4)
def compress_arrow_table(table: pa.Table) -> pa.Buffer:
    options = pa.ipc.IpcWriteOptions(compression='zstd')  # or use 'lz4'
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return sink.getvalue()

def decompress_arrow_table(buffer: pa.Buffer) -> pa.Table:
    reader = pa.ipc.RecordBatchStreamReader(buffer)
    return reader.read_all()

class MyFlightServer(pa.flight.FlightServerBase):
    def do_exchange(self, context, descriptor, reader, writer):
        # Read incoming data (Arrow table)
        arrow_table = reader.read_all()
        
        # Convert to Polars and run the query (example transformation)
        polars_df = pl.from_arrow(arrow_table)
        query_result = run_query(polars_df)  # This is your query function
        
        # Convert the result back to Arrow Table
        result_arrow_table = query_result.to_arrow()
        
        # Compress the result and write back
        compressed_buffer = compress_arrow_table(result_arrow_table)
        writer.write(compressed_buffer)

client = pa.flight.FlightClient.connect('grpc+tcp://localhost:8815')

# Send request and receive the response
flight_descriptor = pa.flight.FlightDescriptor.for_path('my_query')
reader, writer = client.do_exchange(flight_descriptor)

# Write input data (Arrow table) to the server
input_table = ...  # Prepare your Arrow table here
writer.write_table(input_table)

# Read the compressed result from the server
compressed_result = reader.read_all()
result_arrow_table = decompress_arrow_table(compressed_result)

import pyarrow.flight
import json
import pyarrow as pa
import polars as pl

# Client-side
client = pa.flight.FlightClient.connect('grpc+tcp://localhost:8815')

# Create your input data (Arrow table)
input_table = ...  # Prepare your input Arrow table

# Create the JSON metadata to send
json_meta = {
    'query': 'SELECT * FROM table WHERE col > 5',
    'request_id': '12345',
    'additional_info': 'test metadata'
}
json_bytes = json.dumps(json_meta).encode('utf-8')

# FlightDescriptor can also be used to pass metadata in the path or command (optional)
flight_descriptor = pa.flight.FlightDescriptor.for_path('my_query')

# Open the exchange (write input, read output)
reader, writer = client.do_exchange(flight_descriptor)

# Write the Arrow table with JSON metadata
writer.write_table(input_table)
writer.write_metadata(json_bytes)  # Send the metadata after the table

# Close the writer after writing
writer.done_writing()

# Read the result (possibly with metadata returned from the server)
result_arrow_table = reader.read_all()
metadata = reader.read_metadata()  # Read the server's metadata response
if metadata is not None:
    server_meta = json.loads(metadata.to_pybytes())
    print("Server Metadata:", server_meta)

import pyarrow as pa
import pyarrow.flight
import json
import polars as pl

class MyFlightServer(pa.flight.FlightServerBase):
    def do_exchange(self, context, descriptor, reader, writer):
        # Read incoming Arrow table
        arrow_table = reader.read_all()
        
        # Convert to Polars and run the query (this is your transformation logic)
        polars_df = pl.from_arrow(arrow_table)
        query_result = run_query(polars_df)  # Replace with actual query processing logic

        # Convert the result back to an Arrow table
        result_arrow_table = query_result.to_arrow()

        # Read metadata (if available) sent by the client
        client_metadata = reader.read_metadata()
        if client_metadata:
            json_meta = json.loads(client_metadata.to_pybytes())
            print("Received Client Metadata:", json_meta)

            # Optionally, modify behavior based on the metadata
            if json_meta.get('request_id') == '12345':
                print("Special handling for request_id 12345")

        # Prepare result metadata to send back to the client
        response_metadata = {
            'status': 'success',
            'processed_rows': len(result_arrow_table),
            'request_id': json_meta.get('request_id', 'unknown')
        }
        response_metadata_bytes = json.dumps(response_metadata).encode('utf-8')

        # Write the result Arrow table and metadata back to the client
        writer.write_table(result_arrow_table)
        writer.write_metadata(response_metadata_bytes)

