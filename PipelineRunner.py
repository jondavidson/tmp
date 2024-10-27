from typing import Callable, Dict, Any
from datetime import datetime
import os

# Task Runner class to toggle between Dask and immediate execution
class TaskRunner:
    def __init__(self, use_dask: bool = False):
        self.use_dask = use_dask
        if use_dask:
            from dask import delayed
            from dask.distributed import Client, LocalCluster
            self.delayed = delayed
            self.client = Client(LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB'))
        else:
            self.delayed = lambda x: x  # No-op if not using Dask
            self.client = None

    def run_task(self, func: Callable, *args, **kwargs) -> Any:
        task = self.delayed(func)(*args, **kwargs)  # Wrap in @delayed if using Dask
        if self.use_dask:
            future = self.client.compute(task)
            return future.result()
        else:
            return task  # Executes immediately without Dask

    def close(self):
        if self.client:
            self.client.close()

# Example task functions
def generate_data(dataset_name: str, run_date: datetime) -> Dict[str, Any]:
    output_path = f"/data/processed/{dataset_name}_{run_date.strftime('%Y%m%d')}.parquet"
    print(f"Generating data for {dataset_name}, saving to {output_path}")
    return {
        "output_path": output_path,
        "row_count": 10000, 
        "execution_time": 1.2,
        "status": "success",
        "validation": {
            "is_valid": True,
            "summary": "Data generation succeeded"
        }
    }

def process_data(dataset_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    output_path = f"/data/processed/{dataset_name}_processed.parquet"
    print(f"Processing data for {dataset_name} from {input_data['output_path']} to {output_path}")
    return {
        "output_path": output_path,
        "row_count": 9000,
        "execution_time": 2.1,
        "status": "success",
        "validation": {
            "is_valid": True,
            "summary": "Data processing succeeded"
        }
    }

def aggregate_data(dataset_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    output_path = f"/data/processed/{dataset_name}_aggregated.parquet"
    print(f"Aggregating data for {dataset_name} from {input_data['output_path']} to {output_path}")
    return {
        "output_path": output_path,
        "row_count": 8500,
        "execution_time": 1.7,
        "status": "success",
        "validation": {
            "is_valid": True,
            "summary": "Aggregation completed successfully"
        }
    }

# Running the pipeline with TaskRunner
if __name__ == "__main__":
    runner = TaskRunner(use_dask=True)  # Set use_dask=False for immediate execution without Dask

    run_date = datetime.now()
    generated_data = runner.run_task(generate_data, "raw_data", run_date)
    processed_data = runner.run_task(process_data, "processed_data", generated_data)
    aggregated_data = runner.run_task(aggregate_data, "aggregated_data", processed_data)

    print("Pipeline completed with result:", aggregated_data)

    # Close Dask client if used
    runner.close()
