from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
import json
from dask import delayed
from dask.distributed import Client, LocalCluster

# Define Dataset Config Classes
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
        date_str = date.strftime("%Y%m%d")  # File suffix as yyyymmdd
        subdir = date.strftime(self.date_format)
        return os.path.join(self.get_base_dir(), subdir, f"{self.name}_{date_str}.parquet")

class LatestDataset(DatasetConfig):
    def __init__(self, name: str, base_path: str, dataset_type: str, filename: str = "latest.csv", **kwargs):
        super().__init__(name, base_path, dataset_type, **kwargs)
        self.filename = filename

    def get_latest_path(self) -> str:
        return os.path.join(self.get_base_dir(), self.filename)

# Define PipelineConfig to Load and Chain Datasets
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

# Task Functions
@delayed
def generate_data(pipeline_config: PipelineConfig, dataset_name: str, run_date: datetime) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    output_path = dataset.get_path_for_date(run_date) if isinstance(dataset, PartitionedDataset) else dataset.get_latest_path()
    
    # Mock processing logic
    print(f"Generating data for {dataset_name}, saving to {output_path}")
    
    # Return structured metadata for pipeline tracking
    return {
        "output_path": output_path,
        "row_count": 10000,  # Example count
        "execution_time": 1.2,  # Example execution time in seconds
        "status": "success",
        "validation": {
            "is_valid": True,
            "summary": "Data generation succeeded"
        }
    }

@delayed
def process_data(pipeline_config: PipelineConfig, dataset_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    output_path = dataset.get_path_for_date(datetime.now()) if isinstance(dataset, PartitionedDataset) else dataset.get_latest_path()
    
    # Mock processing logic
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

@delayed
def aggregate_data(pipeline_config: PipelineConfig, dataset_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    dataset = pipeline_config.get_dataset_config(dataset_name)
    output_path = dataset.get_latest_path()
    
    # Mock aggregation logic
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

# Pipeline Runner
class PipelineRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.client = Client(LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB'))
        self.config = self.load_config()

    def load_config(self) -> PipelineConfig:
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        return PipelineConfig(base_path=config_data["base_path"], datasets=config_data["datasets"])

    def run(self) -> None:
        # Define tasks with dependencies and chaining
        run_date = datetime.now()
        generated_data = generate_data(self.config, "raw_data", run_date=run_date)
        processed_data = process_data(self.config, "processed_data", input_data=generated_data)
        aggregated_data = aggregate_data(self.config, "aggregated_data", input_data=processed_data)
        
        # Execute the pipeline
        future = self.client.compute(aggregated_data)
        result = future.result()
        print("Pipeline completed with result:", result)

# Example usage
if __name__ == "__main__":
    runner = PipelineRunner("configs/pipeline_config.json")
    runner.run()
