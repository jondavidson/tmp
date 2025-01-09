from datetime import datetime, timedelta
from prefect import flow, task
from prefect.client import get_client

# Task to perform the initial data pull
@task
def initial_data_pull(start_time, end_time):
    print(f"Performing initial data pull: {start_time} to {end_time}")
    # Replace this with actual data-fetching logic
    return True

# Task to perform the incremental data pull
@task
def incremental_data_pull(start_time, end_time):
    print(f"Performing incremental data pull: {start_time} to {end_time}")
    # Replace this with actual data-fetching logic
    return True

# Task to check if the flow should reschedule
@task
def should_reschedule(current_time, end_of_day):
    if current_time <= end_of_day:
        print(f"Rescheduling: Current time {current_time} is within business hours.")
        return True
    print(f"Stopping: Current time {current_time} is outside business hours.")
    return False

@flow
def dynamic_data_pipeline(
    interval: int = 60,  # Interval in seconds for incremental pulls
    start_time: datetime = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
    end_time: datetime = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=0),
):
    current_time = datetime.utcnow()

    # Determine whether to perform the initial or incremental pull
    if current_time == start_time:  # Start of the day
        initial_data_pull(start_time=start_time, end_time=current_time)
    else:  # Incremental pull for the interval period
        incremental_data_pull(
            start_time=current_time - timedelta(seconds=interval),
            end_time=current_time,
        )

    # Reschedule the next run if within business hours
    if should_reschedule(current_time=current_time, end_of_day=end_time):
        client = get_client()
        next_run_time = datetime.utcnow() + timedelta(seconds=interval)  # Next run after interval
        client.create_flow_run_from_name(
            flow_name="dynamic-data-pipeline",
            scheduled_start_time=next_run_time,
            parameters={"interval": interval, "start_time": start_time, "end_time": end_time},
        )


name: dynamic-data-pipeline
flow: dynamic_pipeline.py:dynamic_data_pipeline
schedule:
  cron: "0 0 * * 1-5"  # Trigger the flow at the start of each business day (midnight UTC)
parameters:
  interval: 60  # Interval for incremental pulls (60 seconds)
  start_time: "2025-01-08T00:00:00Z"  # Business day start time (midnight UTC)
  end_time: "2025-01-08T23:59:59Z"  # Business day end time (11:59 PM UTC)
