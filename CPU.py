from dask.distributed import Client, LocalCluster
import os

def set_cpu_affinity():
    cpu_core = int(os.environ.get('DASK_CPU_AFFINITY', -1))
    if cpu_core >= 0:
        os.sched_setaffinity(0, {cpu_core})

# Start a LocalCluster without workers
cluster = LocalCluster(n_workers=0, processes=True)
client = Client(cluster)

num_workers = 8  # Adjust based on your number of cores

for i in range(num_workers):
    # Set an environment variable for each worker
    env = {'DASK_CPU_AFFINITY': str(i)}
    # Start the worker with the environment variable
    cluster.scale(n=i+1, env=env)
    # Register the CPU affinity function
    client.register_worker_callbacks(set_cpu_affinity)
