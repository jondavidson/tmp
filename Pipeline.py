from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json

# Dataset Config and Pipeline Config Classes
class DatasetConfig:
    def __init__(self, name: str, base_path: str, dataset_type: str, inputs: Optional[List[str]] = None):
        self.name = name
        self.base_path = base_path
        self.dataset_type = dataset_type
        self.inputs = inputs or []

    def get_base_dir(self) -> str:
        return os.path.join(self.base_path, self.dataset_type, self.name)

class PartitionedDataset(DatasetConfig):
    def __init__(self, name: str, base_path: str, dataset_type: str, date_format: str = "%Y-%m-%d", **kwargs):
        super().__init__(name, base_path, dataset_type, **kwargs)
        self.date_format = date_format

    def get_path_for_date(self, date: datetime) -> str:
        date_str = date.strftime("%Y%m%d")
        subdir = date.strftime(self.date_format)
        return os.path.join(self.get_base_dir(), subdir, f"{self.name}_{date_str}.parquet")

class LatestDataset(DatasetConfig):
    def __init__(self, name: str, base_path: str, dataset_type: str, filename: str = "latest.csv", **kwargs):
        super().__init__(name, base_path, dataset_type, **kwargs)
        self.filename = filename

    def get_latest_path(self) -> str:
        return os.path.join(self.get_base_dir(), self.filename)

class PipelineConfig:
    def __init__(self, base_path: str, datasets: List[Dict[str, Any]]):
        self.base_path = base_path
        self.datasets = {ds['name']: self._initialize_dataset(ds) for ds in datasets}

    def _initialize_dataset(self, ds: Dict[str, Any]) -> DatasetConfig:
        if ds["type"] == "PartitionedDataset":
            return PartitionedDataset(**ds, base_path=self.base_path)
        elif ds["type"] == "LatestDataset":
            return LatestDataset(**ds, base_path=self.base_path)
        else:
            raise ValueError(f"Unknown dataset type: {ds['type']}")

    def get_dataset_config(self, name: str) -> DatasetConfig:
        return self.datasets.get(name)

    def get_input_paths(self, dataset_name: str, run_date: datetime) -> List[str]:
        """Returns paths of all dependencies for a dataset."""
        dataset = self.get_dataset_config(dataset_name)
        input_paths = []
        for input_name in dataset.inputs:
            input_dataset = self.get_dataset_config(input_name)
            if isinstance(input_dataset, PartitionedDataset):
                input_paths.append(input_dataset.get_path_for_date(run_date))
            elif isinstance(input_dataset, LatestDataset):
                input_paths.append(input_dataset.get_latest_path())
        return input_paths

# Task Runner with Dask or Immediate Execution
class TaskRunner:
    def __init__(self, pipeline_config: PipelineConfig, use_dask: bool = False):
        self.pipeline_config = pipeline_config
        self.use_dask = use_dask
        if use_dask:
            from dask import delayed
            from dask.distributed import Client, LocalCluster
            self.delayed = delayed
            self.client = Client(LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB'))
        else:
            self.delayed = lambda x: x
            self.client = None

    def run_task(self, func: Callable, *args, **kwargs) -> Any:
        task = self.delayed(func)(*args, **kwargs)
        if self.use_dask:
            future = self.client.compute(task)
            return future.result()
        else:
            return task

    def close(self):
        if self.client:
            self.client.close()

# Task Functions
def generate_data(pipeline_config: PipelineConfig, dataset_name: str, run_date: datetime) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    output_path = dataset.get_path_for_date(run_date)
    print(f"Generating data for {dataset_name}, saving to {output_path}")
    return {
        "output_path": output_path,
        "row_count": 10000, 
        "execution_time": 1.2,
        "status": "success",
        "validation": {"is_valid": True, "summary": "Data generation succeeded"}
    }

def process_data(pipeline_config: PipelineConfig, dataset_name: str, run_date: datetime) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    input_paths = pipeline_config.get_input_paths(dataset_name, run_date)
    output_path = dataset.get_path_for_date(run_date)
    
    for input_path in input_paths:
        print(f"Processing {dataset_name} from input {input_path} to output {output_path}")
    
    return {
        "output_path": output_path,
        "row_count": 9000,
        "execution_time": 2.1,
        "status": "success",
        "validation": {"is_valid": True, "summary": "Data processing succeeded"}
    }

def aggregate_data(pipeline_config: PipelineConfig, dataset_name: str, run_date: datetime) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    input_paths = pipeline_config.get_input_paths(dataset_name, run_date)
    output_path = dataset.get_latest_path()
    
    for input_path in input_paths:
        print(f"Aggregating {dataset_name} from input {input_path} to output {output_path}")
    
    return {
        "output_path": output_path,
        "row_count": 8500,
        "execution_time": 1.7,
        "status": "success",
        "validation": {"is_valid": True, "summary": "Aggregation completed successfully"}
    }

# Main Pipeline Execution
if __name__ == "__main__":
    # Load the pipeline configuration from a JSON file
    with open("configs/pipeline_config.json", "r") as f:
        config_data = json.load(f)
    pipeline_config = PipelineConfig(base_path=config_data["base_path"], datasets=config_data["datasets"])

    # Initialize TaskRunner with the pipeline configuration
    runner = TaskRunner(pipeline_config, use_dask=True)

    # Execute tasks with dependencies on multiple inputs
    run_date = datetime.now()
    generated_data = runner.run_task(generate_data, pipeline_config, "raw_data", run_date)
    processed_data = runner.run_task(process_data, pipeline_config, "processed_data", run_date)
    aggregated_data = runner.run_task(aggregate_data, pipeline_config, "aggregated_data", run_date)

    print("Pipeline completed with result:", aggregated_data)

    # Close Dask client if used
    runner.close()
