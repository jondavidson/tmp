import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import argparse
import os

def generate_date_range(start_date: str, end_date: str):
    """
    Generates a list of dates in YYYYMMDD format between start_date and end_date inclusive.
    
    Args:
        start_date (str): Start date in YYYYMMDD format.
        end_date (str): End date in YYYYMMDD format.
    
    Returns:
        List[str]: List of date strings.
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = end - start
    date_list = []
    for i in range(delta.days + 1):
        day = start + timedelta(days=i)
        date_str = day.strftime("%Y%m%d")
        date_list.append(date_str)
    return date_list

def run_script_for_date(script_path: str, date: str, script_args: list, python_executable: str = 'python'):
    """
    Runs the script with the given date and arguments.
    
    Args:
        script_path (str): Path to the script to execute.
        date (str): Date in YYYYMMDD format.
        script_args (list): List of script-specific arguments.
        python_executable (str): Python interpreter to use.
    
    Returns:
        dict: Contains script name, date, stdout, stderr, and return code.
    """
    cmd = [python_executable, script_path, '--sd', date, '--ed', date] + script_args
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {
            'script': script_path,
            'date': date,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.CalledProcessError as e:
        return {
            'script': script_path,
            'date': date,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'returncode': e.returncode
        }

def parallel_execute(
    script_path: str,
    start_date: str,
    end_date: str,
    script_args: list = [],
    max_workers: int = 4,
    python_executable: str = 'python'
):
    """
    Executes the script in parallel across the specified date range.
    
    Args:
        script_path (str): Path to the script to execute.
        start_date (str): Start date in YYYYMMDD format.
        end_date (str): End date in YYYYMMDD format.
        script_args (list, optional): List of script-specific arguments. Defaults to [].
        max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.
        python_executable (str, optional): Python interpreter to use. Defaults to 'python'.
    
    Returns:
        List[dict]: List of result dictionaries for each script execution.
    """
    date_list = generate_date_range(start_date, end_date)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(run_script_for_date, script_path, date, script_args, python_executable): date
            for date in date_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result = future.result()
                results.append(result)
                if result['returncode'] == 0:
                    print(f"✅ [{result['script']}] Completed for date {result['date']}")
                else:
                    print(f"❌ [{result['script']}] Failed for date {result['date']}. Return Code: {result['returncode']}")
                    print(f"--- stderr ---\n{result['stderr']}")
            except Exception as exc:
                print(f"❌ [{script_path}] generated an exception for date {date}: {exc}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Parallel Script Executor Across Dates")
    parser.add_argument('--script', required=True, help='Path to the script to execute')
    parser.add_argument('--start_date', required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--script_args', nargs='*', default=[], help='Script-specific arguments')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--python', default='python', help='Python interpreter to use')
    
    args = parser.parse_args()
    
    # Validate script path
    if not os.path.isfile(args.script):
        print(f"❌ Script not found: {args.script}")
        return
    
    # Generate and execute
    results = parallel_execute(
        script_path=args.script,
        start_date=args.start_date,
        end_date=args.end_date,
        script_args=args.script_args,
        max_workers=args.max_workers,
        python_executable=args.python
    )
    
    # Optionally, handle results further (e.g., logging, aggregation)
    success_count = sum(1 for r in results if r['returncode'] == 0)
    failure_count = len(results) - success_count
    print(f"\n🎯 Execution Summary: {success_count} succeeded, {failure_count} failed out of {len(results)} total runs.")

if __name__ == "__main__":
    main()
