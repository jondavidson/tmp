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


from dask.distributed import Client, SpecCluster, Nanny
import os

def set_cpu_affinity():
    cpu_core = int(os.environ.get('DASK_CPU_AFFINITY', -1))
    if cpu_core >= 0:
        os.sched_setaffinity(0, {cpu_core})

num_workers = 8  # Adjust based on your number of physical cores

worker_specs = {
    f'worker-{i}': {
        'cls': Nanny,
        'options': {
            'env': {'DASK_CPU_AFFINITY': str(i)},
            'plugins': [set_cpu_affinity],
        },
    }
    for i in range(num_workers)
}

cluster = SpecCluster(worker_specs)
client = Client(cluster)

#!/bin/bash
# start_workers_no_contention.sh

SCHEDULER_ADDRESS="tcp://192.168.1.100:8786"
NUM_PHYSICAL_CORES=36  # Number of physical cores

# Set threading environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Start workers pinned to one logical processor per physical core
for i in $(seq 0 $((NUM_PHYSICAL_CORES - 1))); do
    export DASK_CPU_AFFINITY=$i
    taskset -c $i dask-worker $SCHEDULER_ADDRESS --nthreads 1 --memory-limit 0 &
done
