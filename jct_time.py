import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator

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

# Function to generate jobs
def generate_jobs(num_jobs, start_time=0, time_interval=5):
    jobs = []
    for i in range(num_jobs):
        submission_time = start_time + i * time_interval
        duration = random.randint(5, 20)
        num_param_servers = 1
        num_workers = random.randint(2, 5)
        cpu_req = random.randint(2, 4)
        gpu_req = random.randint(1, 3)
        bandwidth_req = random.randint(5, 20)
        jobs.append(Job(submission_time, duration, num_param_servers, num_workers, cpu_req, gpu_req, bandwidth_req))
    return jobs

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
                print(f"Job placed at time {current_time}: {job.placement}")

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
                print(f"Job placed at time {current_time}: {job.placement}")

# DRF Placement algorithm
def place_jobs_drf(cluster, jobs, current_time):
    for job in jobs:
        if job.submission_time <= current_time and job.start_time is None:
            ps_remaining = job.num_param_servers
            workers_remaining = job.num_workers
            placement = []
            
            # Determine the dominant resource for the job
            if job.cpu_req >= job.gpu_req:
                dominant_resource = 'cpu'
            else:
                dominant_resource = 'gpu'
            
            # Sort nodes by the dominant resource availability in descending order
            if dominant_resource == 'cpu':
                sorted_nodes = sorted(range(cluster.num_nodes), key=lambda x: cluster.available_cpus[x], reverse=True)
            else:
                sorted_nodes = sorted(range(cluster.num_nodes), key=lambda x: cluster.available_gpus[x], reverse=True)

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
                print(f"Job placed at time {current_time}: {job.placement}")

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
    gpu_utilization = []
    bandwidth_utilization = []
    
    while jobs or jobs_in_progress:
        # Add jobs to progress
        placement_algorithm(cluster, jobs, current_time)
        for job in jobs:
            if job.start_time is not None and job not in jobs_in_progress:
                jobs_in_progress.append(job)
                jobs.remove(job)
        
        # Calculate penalties for bandwidth bottlenecks
        penalties = calculate_bandwidth_penalty(cluster, jobs_in_progress)
        for job, penalty in penalties.items():
            job.adjusted_duration = job.base_duration * (1 + penalty)
            job.end_time = job.start_time + job.adjusted_duration

        # Track GPU and bandwidth utilization
        current_gpu_utilization = sum(cluster.gpus_per_node - g for g in cluster.available_gpus)
        current_bandwidth_utilization = sum(sum(cluster.bandwidth_matrix[node]) for node in range(cluster.num_nodes))
        gpu_utilization.append((current_time, current_gpu_utilization))
        bandwidth_utilization.append((current_time, current_bandwidth_utilization))

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
                print(f"Job completed at time {current_time}: {job.placement}")
        else:
            current_time += 1

        # Debugging: Print current time and number of jobs in progress
        print(f"Current time: {current_time}, Jobs in progress: {len(jobs_in_progress)}, Jobs remaining: {len(jobs)}")
    
    job_completion_times = [(job.submission_time, job.start_time, job.end_time) for job in all_jobs]
    makespan = max(job.end_time for job in all_jobs) if all_jobs else current_time
    return job_completion_times, makespan, gpu_utilization, bandwidth_utilization

# Calculate average JCT
def calculate_avg_jct(job_completion_times, jobs):
    normalized_jcts = []

    for (submission_time, start_time, end_time), job in zip(job_completion_times, jobs):
        jct = float(end_time) - submission_time
        fixed_duration = job.base_duration
        normalized_jct = jct / fixed_duration
        normalized_jcts.append(normalized_jct)
    
    avg_normalized_jct = np.mean(normalized_jcts) if normalized_jcts else float('nan')
    return avg_normalized_jct, normalized_jcts

# Pad shorter sequences with their last value to ensure all sequences have the same length
def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        last_value = seq[-1][1] if seq else 0
        while len(seq) < max_length:
            seq.append((seq[-1][0] + 1, last_value))
        padded_sequences.append(seq)
    return padded_sequences

# Run multiple simulations
num_simulations = 100
num_nodes = 3
gpus_per_node = 3
cpus_per_node = 4
bandwidth_matrix = np.array([[0, 10, 10], [10, 0, 10], [100, 100, 0]])

num_jobs = 6  # Set the number of jobs
jobs = generate_jobs(num_jobs)

avg_jct_results = {"original": [], "modified": [], "drf": []}
makespan_results = {"original": [], "modified": [], "drf": []}
gpu_utilization_results = {"original": [], "modified": [], "drf": []}
bandwidth_utilization_results = {"original": [], "modified": [], "drf": []}

for sim in range(num_simulations):
    print(f"Simulation {sim+1}/{num_simulations}")
    # Initialize clusters for all simulators
    cluster_original = Cluster(num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix)
    cluster_modified = Cluster(num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix)
    cluster_drf = Cluster(num_nodes, gpus_per_node, cpus_per_node, bandwidth_matrix)

    # Initialize jobs
    jobs_list_original = copy.deepcopy(jobs)
    jobs_list_modified = copy.deepcopy(jobs)
    jobs_list_drf = copy.deepcopy(jobs)

    # Run simulations for all algorithms
    job_completion_times_original, makespan_original, gpu_utilization_original, bandwidth_utilization_original = simulate(cluster_original, jobs_list_original, place_jobs_original)
    job_completion_times_modified, makespan_modified, gpu_utilization_modified, bandwidth_utilization_modified = simulate(cluster_modified, jobs_list_modified, place_jobs_modified)
    job_completion_times_drf, makespan_drf, gpu_utilization_drf, bandwidth_utilization_drf = simulate(cluster_drf, jobs_list_drf, place_jobs_drf)

    # Calculate average JCT for all algorithms
    avg_jct_percentage_original, _ = calculate_avg_jct(job_completion_times_original, jobs)
    avg_jct_percentage_modified, _ = calculate_avg_jct(job_completion_times_modified, jobs)
    avg_jct_percentage_drf, _ = calculate_avg_jct(job_completion_times_drf, jobs)

    # Store results
    avg_jct_results["original"].append(avg_jct_percentage_original)
    avg_jct_results["modified"].append(avg_jct_percentage_modified)
    avg_jct_results["drf"].append(avg_jct_percentage_drf)

    makespan_results["original"].append(makespan_original)
    makespan_results["modified"].append(makespan_modified)
    makespan_results["drf"].append(makespan_drf)
    
    gpu_utilization_results["original"].append(gpu_utilization_original)
    gpu_utilization_results["modified"].append(gpu_utilization_modified)
    gpu_utilization_results["drf"].append(gpu_utilization_drf)
    
    bandwidth_utilization_results["original"].append(bandwidth_utilization_original)
    bandwidth_utilization_results["modified"].append(bandwidth_utilization_modified)
    bandwidth_utilization_results["drf"].append(bandwidth_utilization_drf)

# Pad sequences
for algo in gpu_utilization_results:
    gpu_utilization_results[algo] = pad_sequences(gpu_utilization_results[algo])
for algo in bandwidth_utilization_results:
    bandwidth_utilization_results[algo] = pad_sequences(bandwidth_utilization_results[algo])

# Calculate means and confidence intervals
def mean_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, h

avg_jct_mean_ci = {k: mean_confidence_interval(v) for k, v in avg_jct_results.items()}
makespan_mean_ci = {k: mean_confidence_interval(v) for k, v in makespan_results.items()}

# Plot the results side by side with confidence intervals
def set_integer_ylabels(ax):
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Plot average JCT with confidence intervals
plt.figure(figsize=(5, 3))
labels_y = ['DRF', 'Modified', 'Original']
bar_colors_jct = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom colors

means = [avg_jct_mean_ci[key][0] for key in ["drf", "modified", "original"]]
errors = [avg_jct_mean_ci[key][1] for key in ["drf", "modified", "original"]]

barlist = plt.bar(labels_y, means, yerr=errors, color=bar_colors_jct, capsize=5, width=0.6)
for bar in barlist:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
plt.ylabel('Normalized JCT')
ax1 = plt.gca()
set_integer_ylabels(ax1)
plt.tight_layout()

plt.savefig('average_jct_ci.png')
plt.show()

# Plot makespan with confidence intervals
plt.figure(figsize=(5, 3))
bar_colors_makespan = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom colors

means = [makespan_mean_ci[key][0] for key in ["drf", "modified", "original"]]
errors = [makespan_mean_ci[key][1] for key in ["drf", "modified", "original"]]

barlist2 = plt.bar(labels_y, means, yerr=errors, color=bar_colors_makespan, capsize=5, width=0.6)
for bar in barlist2:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
plt.ylabel('Normalized Makespan')
plt.tight_layout()
ax2 = plt.gca()
set_integer_ylabels(ax2)
plt.savefig('makespan_ci.png')
plt.show()

# Plot GPU and Bandwidth utilization
def plot_utilization_over_time(utilization_data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for algo in utilization_data:
        time_series = np.mean([np.array(data)[:, 1] for data in utilization_data[algo]], axis=0)
        plt.plot(np.array(utilization_data[algo][0])[:, 0], time_series, label=algo)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

plot_utilization_over_time(gpu_utilization_results, 'GPU Utilization Over Time', 'GPU Utilization', 'gpu_utilization.png')
plot_utilization_over_time(bandwidth_utilization_results, 'Bandwidth Utilization Over Time', 'Bandwidth Utilization', 'bandwidth_utilization.png')
