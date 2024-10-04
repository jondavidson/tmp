import argparse
import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Set, Tuple

import subprocess
import yaml
from prometheus_client import Summary, Counter, start_http_server
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn

# Define metrics for monitoring
SCRIPT_EXECUTION_TIME = Summary('script_execution_time_seconds', 'Time spent executing script', ['script_name'])
SCRIPT_FAILURES = Counter('script_failures_total', 'Total number of script failures', ['script_name'])


def retry(max_retries: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Attempt {attempt} failed with error: {e}")
                    if attempt == max_retries:
                        logging.error(f"All {max_retries} retries failed for function {func.__name__}")
                        raise
        return wrapper
    return decorator


class Dataset:
    def __init__(self, name: str, dataset_type: str, location: str, date_format: Optional[str] = None) -> None:
        self.name: str = name
        self.type: str = dataset_type  # 'latest' or 'dated'
        self.location: str = location
        self.date_format: Optional[str] = date_format

    def resolve_path(self, context: Dict[str, Any]) -> str:
        if self.type == 'dated':
            return self.location.format(**context)
        return self.location


class Script:
    def __init__(
        self,
        name: str,
        outputs: List[str],
        inputs: List[str],
        args: Optional[List[str]] = None,
        date_parameters: bool = False
    ) -> None:
        self.name: str = name
        self.outputs: List[str] = outputs  # List of output dataset names
        self.inputs: List[str] = inputs    # List of input dataset names
        self.args: List[str] = args or []
        self.date_parameters: bool = date_parameters
        self.dependencies: List['Script'] = []  # Scripts that this script depends on
        self.dependents: List['Script'] = []    # Scripts that depend on this script

    def add_dependency(self, script: 'Script') -> None:
        self.dependencies.append(script)

    def add_dependent(self, script: 'Script') -> None:
        self.dependents.append(script)

    def is_ready(self, completed_scripts: Set[str]) -> bool:
        return all(dep.name in completed_scripts for dep in self.dependencies)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Script):
            return self.name == other.name
        return False

    @retry(max_retries=3)
    def execute(self, context: Dict[str, Any]) -> bool:
        with SCRIPT_EXECUTION_TIME.labels(script_name=self.name).time():
            # Resolve arguments with placeholders
            args_with_values = self._resolve_arguments(context)
            # Add date parameters if required
            if self.date_parameters:
                args_with_values.extend(['--start_date', context['start_date'], '--end_date', context['end_date']])
            # Build the command
            command = ['python', self.name] + args_with_values
            logging.info(f"Starting {self.name} with arguments: {args_with_values}")
            try:
                subprocess.run(command, check=True, timeout=3600)
                logging.info(f"Finished {self.name}")
                return True
            except subprocess.CalledProcessError as e:
                logging.error(f"Script {self.name} failed with error: {e}")
                SCRIPT_FAILURES.labels(script_name=self.name).inc()
                return False
            except subprocess.TimeoutExpired as e:
                logging.error(f"Script {self.name} timed out: {e}")
                SCRIPT_FAILURES.labels(script_name=self.name).inc()
                return False

    def _resolve_arguments(self, context: Dict[str, Any]) -> List[str]:
        # Placeholder substitution in arguments
        return [arg.format(**context) if '{' in arg else arg for arg in self.args]


class DependencyGraph:
    def __init__(self, scripts: List[Script]) -> None:
        self.scripts: Dict[str, Script] = {script.name: script for script in scripts}
        self.build_graph()

    def build_graph(self) -> None:
        # Build dependencies based on inputs and outputs
        dataset_to_script: Dict[str, Script] = {}
        for script in self.scripts.values():
            for output in script.outputs:
                dataset_to_script[output] = script

        for script in self.scripts.values():
            for dataset_name in script.inputs:
                if dataset_name in dataset_to_script:
                    dependency_script = dataset_to_script[dataset_name]
                    script.add_dependency(dependency_script)
                    dependency_script.add_dependent(script)
                else:
                    # External dataset; no action needed
                    pass

        # Check for cycles
        self.detect_cycles()

    def detect_cycles(self) -> None:
        visited: Set[Script] = set()
        stack: Set[Script] = set()

        def visit(script: Script) -> None:
            if script in stack:
                raise Exception(f"Cycle detected in the dependency graph at '{script.name}'.")
            if script in visited:
                return
            stack.add(script)
            for dep in script.dependencies:
                visit(dep)
            stack.remove(script)
            visited.add(script)

        for script in self.scripts.values():
            if script not in visited:
                visit(script)


class PipelineExecutor:
    def __init__(self, scripts: Dict[str, Script], datasets: Dict[str, Dataset], max_workers: int = 4) -> None:
        self.scripts: Dict[str, Script] = scripts
        self.datasets: Dict[str, Dataset] = datasets
        self.max_workers: int = max_workers
        self.completed_scripts: Set[str] = set()
        self.in_progress_scripts: Set[str] = set()
        self.lock = threading.Lock()
        self.progress: Dict[str, str] = {}
        self.console = Console()

    def execute(self, context: Dict[str, Any]) -> None:
        # Initialize ready scripts
        ready_scripts = [script for script in self.scripts.values() if not script.dependencies]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            # Submit initial scripts
            for script in ready_scripts:
                future = executor.submit(self._run_script, script, context)
                futures[future] = script

            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Executing pipeline...", total=len(self.scripts))
                while futures:
                    # Wait for any script to complete
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        script = futures.pop(future)
                        success = future.result()
                        with self.lock:
                            if success:
                                self.completed_scripts.add(script.name)
                                self.progress[script.name] = 'Completed'
                            else:
                                self.progress[script.name] = 'Failed'
                            progress.update(task, advance=1)
                        # Schedule dependent scripts if ready
                        for dependent in script.dependents:
                            if dependent.is_ready(self.completed_scripts) and dependent.name not in self.in_progress_scripts:
                                future_dep = executor.submit(self._run_script, dependent, context)
                                futures[future_dep] = dependent

    def _run_script(self, script: Script, context: Dict[str, Any]) -> bool:
        with self.lock:
            self.in_progress_scripts.add(script.name)
            self.progress[script.name] = 'Running'
        success = script.execute(context)
        with self.lock:
            self.in_progress_scripts.remove(script.name)
        return success


def parse_config(config_path: str) -> Tuple[Dict[str, Dataset], List[Script]]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Create Dataset instances
    datasets: Dict[str, Dataset] = {}
    for name, data in config.get('datasets', {}).items():
        datasets[name] = Dataset(
            name=name,
            dataset_type=data['type'],
            location=data['location'],
            date_format=data.get('date_format')
        )

    # Create Script instances
    scripts: List[Script] = []
    for script_data in config['scripts']:
        scripts.append(Script(
            name=script_data['name'],
            outputs=script_data.get('outputs', []),
            inputs=script_data.get('inputs', []),
            args=script_data.get('args', []),
            date_parameters=script_data.get('date_parameters', False)
        ))

    return datasets, scripts


def parse_dates() -> Tuple[datetime.date, datetime.date]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default=None)
    parser.add_argument('--end_date', type=str, default=None)
    args = parser.parse_args()

    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        start_date = datetime.date.today() - datetime.timedelta(days=1)  # Default to yesterday

    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = start_date

    if end_date < start_date:
        raise ValueError("end_date cannot be earlier than start_date")

    return start_date, end_date


def main() -> None:
    # Start Prometheus metrics server
    start_http_server(8000)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )

    # Parse configuration and dates
    datasets, script_list = parse_config('pipeline_config.yaml')
    start_date, end_date = parse_dates()

    # Build the dependency graph
    scripts_dict = {script.name: script for script in script_list}
    dependency_graph = DependencyGraph(script_list)

    # Prepare context
    date_list = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    for current_date in date_list:
        logging.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        date_str = current_date.strftime('%Y%m%d')
        context: Dict[str, Any] = {
            'date': date_str,
            'start_date': current_date.strftime('%Y-%m-%d'),
            'end_date': current_date.strftime('%Y-%m-%d'),
        }

        # Update context with dataset paths
        for dataset_name, dataset in datasets.items():
            dataset_path = dataset.resolve_path(context)
            context[dataset_name] = dataset_path

        executor = PipelineExecutor(scripts_dict, datasets)
        executor.execute(context)
        # Log the progress after each date
        logging.info(f"Progress for date {current_date.strftime('%Y-%m-%d')}: {executor.progress}")


if __name__ == '__main__':
    main()
