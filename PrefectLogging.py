import os
import subprocess
from prefect import task, flow, get_run_logger
from prefect.task_runners import DaskTaskRunner


# A factory function to create a dynamically named task
def create_script_task(script_name):
    @task(name=f"Run Script: {script_name}")
    def run_script_with_logging(env_vars=None):
        """
        Runs a script using subprocess.Popen, captures logging output,
        and integrates it into Prefect's logging system.

        Args:
            env_vars (dict): Optional dictionary of environment variables to set for the script.

        Returns:
            str: Completion message.
        """
        # Get Prefect logger
        logger = get_run_logger()

        # Use the current environment and update with the provided env_vars
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        logger.info(f"Starting script {script_name} with environment: {env_vars}")

        # Run the script and capture both stdout and stderr
        process = subprocess.Popen(
            ["python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,  # Ensures output is captured as text (not bytes)
        )

        # Stream and log the output line by line
        for line in iter(process.stdout.readline, ""):
            logger.info(f"[{script_name} STDOUT]: {line.strip()}")

        for line in iter(process.stderr.readline, ""):
            logger.error(f"[{script_name} STDERR]: {line.strip()}")

        # Wait for the process to complete
        process.wait()

        # Check for errors
        if process.returncode != 0:
            raise RuntimeError(f"Script {script_name} failed with return code {process.returncode}")

        logger.info(f"Script {script_name} completed successfully.")
        return f"{script_name} completed"

    return run_script_with_logging


# Prefect flow to orchestrate execution
@flow(task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": 4, "threads_per_worker": 1}))
def script_execution_flow(scripts_to_run, global_env=None):
    """
    Orchestrates the execution of scripts based on their dependencies, with dynamic task naming.

    Args:
        scripts_to_run (list): List of scripts with metadata, including environment variables.
        global_env (dict): Optional global environment variables for all scripts.
    """
    logger = get_run_logger()

    # Default global environment
    global_env = global_env or {}

    # Execute scripts
    for script in scripts_to_run:
        # Extract script-specific environment variables
        env_vars = script.get("env", {})
        # Merge global and script-specific environment variables
        env_vars.update(global_env)

        # Create a dynamically named task
        script_task = create_script_task(script["name"])
        logger.info(f"Submitting {script['name']} for execution.")
        script_task.submit(env_vars)

    logger.info("All scripts executed successfully!")
