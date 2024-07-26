import copy
import numpy as np
import matplotlib.pyplot as plt

# Cluster configuration
class Cluster:
    def __init__(self, num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.cpus_per_node = cpus_per_node
        self.bandwidth_matrix = bandwidth_matrix
        self.available_gpus = [gpus_per_node] * num_nodes
        self.available_cpus = [cpus_per_node] * num_nodes

# Job configuration
class Job:
    def __init__(self, submission_time, duration, num_param_servers, num_workers, cpu_req, gpu_req, bandwidth_req):
        self.submission_time = submission_time
        self.base_duration = duration
        self.num_param_servers = num_param_servers
        self.num_workers = num_workers
        self.cpu_req = cpu_req
        self.gpu_req = gpu_req
        self.bandwidth_req = bandwidth_req
        self.start_time = None
        self.end_time = None
        self.placement = None
        self.adjusted_duration = duration

# Original Placement algorithm
def place_jobs_original(cluster, jobs, current_time):
    for job in jobs:
        if job.submission_time <= current_time and job.start_time is None:
            ps_remaining = job.num_param_servers
            workers_remaining = job.num_workers
            placement = []
            for node in range(cluster.num_nodes):
                if ps_remaining <= 0 and workers_remaining <= 0:
                    break
                gpus_available = cluster.available_gpus[node]
                cpus_available = cluster.available_cpus[node]
                while gpus_available > 0 and cpus_available > 0 and (ps_remaining > 0 or workers_remaining > 0):
                    if ps_remaining > 0:
                        cluster.available_gpus[node] -= job.gpu_req / (job.num_param_servers + job.num_workers)
                        cluster.available_cpus[node] -= job.cpu_req / (job.num_param_servers + job.num_workers)
                        gpus_available -= 1
                        cpus_available -= 1
                        ps_remaining -= 1
                        placement.append((node, 'ps'))
                    elif workers_remaining > 0:
                        cluster.available_gpus[node] -= job.gpu_req / (job.num_param_servers + job.num_workers)
                        cluster.available_cpus[node] -= job.cpu_req / (job.num_param_servers + job.num_workers)
                        gpus_available -= 1
                        cpus_available -= 1
                        workers_remaining -= 1
                        placement.append((node, 'worker'))
            if ps_remaining <= 0 and workers_remaining <= 0:
                job.start_time = current_time
                job.placement = placement

# Modified Placement algorithm
def place_jobs_modified(cluster, jobs, current_time):
    for job in jobs:
        if job.submission_time <= current_time and job.start_time is None:
            ps_remaining = job.num_param_servers
            workers_remaining = job.num_workers
            placement = []
            
            # Sort nodes by their total outgoing bandwidth capacity in descending order
            sorted_nodes = sorted(range(cluster.num_nodes), key=lambda x: np.sum(cluster.bandwidth_matrix[x]), reverse=True)

            for node in sorted_nodes:
                if ps_remaining <= 0 and workers_remaining <= 0:
                    break
                gpus_available = cluster.available_gpus[node]
                cpus_available = cluster.available_cpus[node]
                while gpus_available > 0 and cpus_available > 0 and (ps_remaining > 0 or workers_remaining > 0):
                    if ps_remaining > 0:
                        cluster.available_gpus[node] -= job.gpu_req / (job.num_param_servers + job.num_workers)
                        cluster.available_cpus[node] -= job.cpu_req / (job.num_param_servers + job.num_workers)
                        gpus_available -= 1
                        cpus_available -= 1
                        ps_remaining -= 1
                        placement.append((node, 'ps'))
                    elif workers_remaining > 0:
                        cluster.available_gpus[node] -= job.gpu_req / (job.num_param_servers + job.num_workers)
                        cluster.available_cpus[node] -= job.cpu_req / (job.num_param_servers + job.num_workers)
                        gpus_available -= 1
                        cpus_available -= 1
                        workers_remaining -= 1
                        placement.append((node, 'worker'))
            if ps_remaining <= 0 and workers_remaining <= 0:
                job.start_time = current_time
                job.placement = placement

# Calculate bandwidth penalties
def calculate_bandwidth_penalty(cluster, jobs_in_progress):
    total_bandwidth_usage = np.zeros((cluster.num_nodes, cluster.num_nodes))
    penalties = {}

    for job in jobs_in_progress:
        ps_nodes = [node for node, role in job.placement if role == 'ps']
        worker_nodes = [node for node, role in job.placement if role == 'worker']

        for ps_node in ps_nodes:
            for worker_node in worker_nodes:
                if ps_node != worker_node:
                    total_bandwidth_usage[ps_node][worker_node] += job.bandwidth_req

    for job in jobs_in_progress:
        ps_nodes = [node for node, role in job.placement if role == 'ps']
        worker_nodes = [node for node, role in job.placement if role == 'worker']
        max_penalty = 0

        for ps_node in ps_nodes:
            for worker_node in worker_nodes:
                if ps_node != worker_node:
                    available_bandwidth = cluster.bandwidth_matrix[ps_node][worker_node]
                    if total_bandwidth_usage[ps_node][worker_node] > available_bandwidth:
                        penalty = total_bandwidth_usage[ps_node][worker_node] / available_bandwidth
                        max_penalty = max(max_penalty, penalty)
        
        penalties[job] = max_penalty

    return penalties

# Simulation
def simulate(cluster, jobs, placement_algorithm):
    current_time = 0
    jobs_in_progress = []
    all_jobs = jobs[:]
    
    while jobs or jobs_in_progress:
        # Add jobs to progress
        placement_algorithm(cluster, jobs, current_time)
        for job in jobs:
            if job.start_time is not None and job not in jobs_in_progress:
                jobs_in_progress.append(job)
                jobs.remove(job)
                print(job.placement)
        
        # Calculate penalties for bandwidth bottlenecks
        penalties = calculate_bandwidth_penalty(cluster, jobs_in_progress)
        for job, penalty in penalties.items():
            job.adjusted_duration = job.base_duration * (1 + penalty)
            job.end_time = job.start_time + job.adjusted_duration

        # Move time to the next event
        if jobs_in_progress:
            next_event_time = min(job.end_time for job in jobs_in_progress)
            current_time = next_event_time
            
            # Update the resources and remove completed jobs
            completed_jobs = [job for job in jobs_in_progress if job.end_time == current_time]
            for job in completed_jobs:
                jobs_in_progress.remove(job)
                for node, role in job.placement:
                    cluster.available_gpus[node] += job.gpu_req / (job.num_param_servers + job.num_workers)
                    cluster.available_cpus[node] += job.cpu_req / (job.num_param_servers + job.num_workers)
        else:
            current_time += 1
    
    job_completion_times = [(job.submission_time, job.start_time, job.end_time) for job in all_jobs]
    makespan = max(job.end_time for job in all_jobs) if all_jobs else current_time
    return job_completion_times, makespan

def calculate_avg_jct(job_completion_times, jobs):
    jct_percentages = []

    for (submission_time, start_time, end_time), job in zip(job_completion_times, jobs):
        jct = float(end_time) - submission_time
        fixed_duration = job.base_duration
        jct_percentage = (jct / fixed_duration) * 100
        jct_percentages.append(jct_percentage)
    
    print("JCT Percentages:", jct_percentages)
    avg_jct_percentage = np.mean(jct_percentages) if jct_percentages else float('nan')
    return avg_jct_percentage, jct_percentages

# Example usage
num_nodes = 3
gpus_per_node = 3
cpus_per_node = 4
bandwidth_matrix = np.array([[0, 10, 10], [10, 0, 10], [100, 100, 0]])

# Initialize clusters for both simulators
cluster_original = Cluster(num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix)
cluster_modified = Cluster(num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix)

# Initialize jobs
jobs = [
    Job(submission_time=0, duration=10, num_param_servers=1, num_workers=4, cpu_req=3, gpu_req=2, bandwidth_req=10),
    Job(submission_time=0, duration=10, num_param_servers=1, num_workers=3, cpu_req=4, gpu_req=3, bandwidth_req=5),
    Job(submission_time=0, duration=8, num_param_servers=1, num_workers=2, cpu_req=2, gpu_req=1, bandwidth_req=20)
]

jobs_list_original = copy.deepcopy(jobs)
jobs_list_modified = copy.deepcopy(jobs)

# Run simulations for both algorithms
job_completion_times_original, makespan_original = simulate(cluster_original, jobs_list_original, place_jobs_original)
job_completion_times_modified, makespan_modified = simulate(cluster_modified, jobs_list_modified, place_jobs_modified)

# Calculate average JCT for both algorithms
avg_jct_percentage_original, jct_percentages_original = calculate_avg_jct(job_completion_times_original, jobs)
avg_jct_percentage_modified, jct_percentages_modified = calculate_avg_jct(job_completion_times_modified, jobs)

# Plot the results side by side
plt.figure(figsize=(14, 6))

# Plot average JCT
plt.subplot(1, 2, 1)
plt.bar(['Original', 'Modified'], [avg_jct_percentage_original, avg_jct_percentage_modified], color=['blue', 'green'])
plt.ylabel('Percentage (%)')
plt.title('Average JCT in Percentage')

# Plot makespan
plt.subplot(1, 2, 2)
plt.bar(['Original', 'Modified'], [makespan_original, makespan_modified], color=['blue', 'green'])
plt.ylabel('Time')
plt.title('Makespan')

plt.tight_layout()
plt.savefig('comparison_results.png')
plt.savefig('jct.png')

# Print calculated values
print("Original Average JCT Percentage:", avg_jct_percentage_original)
print("Modified Average JCT Percentage:", avg_jct_percentage_modified)
print("Original Makespan:", makespan_original)
print("Modified Makespan:", makespan_modified)