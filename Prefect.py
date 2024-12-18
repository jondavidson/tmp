import os
import subprocess
from prefect import task, flow, get_run_logger
from prefect.task_runners import DaskTaskRunner


# Prefect task to run scripts with logging
@task
def run_script_with_logging(script_name, env_vars=None):
    """
    Runs a script using subprocess.Popen, captures its logging output, 
    and integrates it into Prefect's logging system.

    Args:
        script_name (str): Path to the script to execute.
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

    # Run the script and capture stdout and stderr
    process = subprocess.Popen(
        ["python", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,  # Ensures output is captured as text (str) instead of bytes
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


# Prefect flow to orchestrate execution
@flow(task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": 4, "threads_per_worker": 1}))
def script_execution_flow(scripts_to_run, global_env=None):
    """
    Orchestrates the execution of scripts based on their dependencies, with logging.

    Args:
        scripts_to_run (list): List of scripts with metadata.
        global_env (dict): Optional global environment variables for all scripts.
    """
    logger = get_run_logger()

    # Default global environment
    global_env = global_env or {}

    # Validate and execute scripts
    for script in scripts_to_run:
        env_vars = script.get("env", {})
        env_vars.update(global_env)  # Merge script-specific and global environment
        logger.info(f"Submitting {script['name']} for execution.")
        run_script_with_logging.submit(script["name"], env_vars)

    logger.info("All scripts executed successfully!")


# Example script list with environment variables
scripts_to_run = [
    {"name": "script_1.py", "type": "independent", "env": {"API_KEY": "12345"}},
    {"name": "script_2.py", "type": "independent", "env": {"DEBUG": "1"}},
    {"name": "script_3.py", "type": "dependent", "dependency": "script_1.py"},
    {"name": "script_4.py", "type": "dependent", "dependency": "script_3.py"},
]

# Global environment variables (optional)
global_env = {"GLOBAL_SETTING": "enabled"}

# Run the flow
if __name__ == "__main__":
    script_execution_flow(scripts_to_run, global_env)
