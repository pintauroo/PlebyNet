import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy import stats

# ------------------------------
# 1. Setup and Configuration
# ------------------------------

# Define the directory containing your CSV files
data_directory = '.'  # Adjust the path if necessary

# Define the directory to save plots
plots_directory = 'plots'
os.makedirs(plots_directory, exist_ok=True)

# Define the filename pattern for main files and allocations files
# Main files: 100J_80N_NFD_NHN_NDJ_NBW_rep_utility_FIFO.csv
# Allocations files: 100J_80N_NFD_NHN_NDJ_NBW_rep_utility_FIFO_allocations.csv
main_file_regex = re.compile(r'50J_50N_NFD_HN_NDJ_BW_(\d+)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv')
# main_file_regex = re.compile(r'150J_100N_NFD_HN_NDJ_NBW_(\d+)_(TETRIS|DRF)_FIFO\.csv')
alloc_file_suffix = '_jobs_report.csv'

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
# selected_utilities = ['TETRIS', 'DRF']
selected_reps = range(1, 50)  # Replications 1 to 49

# List all main CSV files matching the pattern
file_list = [f for f in os.listdir(data_directory) if main_file_regex.match(f)]

if not file_list:
    print("No main CSV files found matching the pattern in the specified directory.")
    exit(1)

print(f"Found {len(file_list)} main files to process.")

# ------------------------------
# 2. Define Utility Functions
# ------------------------------

def compute_jains_index(x):
    """
    Compute Jain's Fairness Index for a list of utilization ratios.
    """
    x = np.array(x)
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x ** 2)
    n = len(x)
    return (sum_x ** 2) / (n * sum_x_sq) if sum_x_sq != 0 else 0

def compute_ginis_index(x):
    """
    Compute Gini's Index for a list of utilization ratios.
    """
    x = np.array(x)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumulative_x = np.cumsum(x_sorted)
    sum_x = cumulative_x[-1]
    if sum_x == 0:
        return 0
    gini = (2 * np.sum((np.arange(1, n + 1) * x_sorted))) / (n * sum_x) - (n + 1) / n
    return gini

def compute_throughput(jobs_df, current_time):
    """
    Compute the number of jobs completed at the current_time.
    """
    if jobs_df is None or jobs_df.empty:
        return 0
    completed_jobs = jobs_df[jobs_df['complete_time'] == current_time]
    return len(completed_jobs)

def compute_job_latency(jobs_df, current_time):
    """
    Compute the average job latency for jobs completed at the current_time.
    """
    if jobs_df is None or jobs_df.empty:
        return 0
    completed_jobs = jobs_df[jobs_df['complete_time'] == current_time]
    if completed_jobs.empty:
        return 0
    latencies = completed_jobs['complete_time'] - completed_jobs['submit_time']
    return latencies.mean()

def compute_queue_length(jobs_df, current_time):
    """
    Compute the number of jobs waiting in the queue at the current_time.
    """
    if jobs_df is None or jobs_df.empty:
        return 0
    pending_jobs = jobs_df[(jobs_df['submit_time'] <= current_time) & (jobs_df['complete_time'] > current_time)]
    return len(pending_jobs)

def compute_active_jobs(jobs_df, current_time):
    """
    Compute the number of active jobs being processed at the current_time.
    """
    if jobs_df is None or jobs_df.empty:
        return 0
    active_jobs = jobs_df[(jobs_df['submit_time'] <= current_time) & (jobs_df['complete_time'] > current_time)]
    return len(active_jobs)

# ------------------------------
# 3. Initialize Master DataFrame
# ------------------------------

# Columns for the master DataFrame
master_columns = [
    'Utility_Function',
    'Replication',
    'Quartile',
    'Jains_GPU_Fairness',
    'Ginis_GPU_Index',
    'Average_GPU_Utilization',
    'Throughput_Jobs_Completed',
    'Average_Job_Latency',
    'Queue_Length',
    'Active_Jobs'
]

master_metrics = []

# ------------------------------
# 4. Process Each File
# ------------------------------

for filename in file_list:
    match = main_file_regex.match(filename)
    if not match:
        print(f"Filename '{filename}' does not match the expected pattern. Skipping.")
        continue

    rep = int(match.group(1))
    utility_function = match.group(2)

    # Filter based on selected utilities and replications
    if utility_function not in selected_utilities or rep not in selected_reps:
        print(f"Skipping file '{filename}' as it does not match selected utilities or replications.")
        continue

    filepath = os.path.join(data_directory, filename)

    # Determine the corresponding allocations file
    alloc_filename = filename.replace('.csv', alloc_file_suffix)
    alloc_filepath = os.path.join(data_directory, alloc_filename)

    if not os.path.exists(alloc_filepath):
        print(f"Allocations file '{alloc_filename}' not found for '{filename}'. Skipping.")
        continue

    # Load the main CSV file
    try:
        df_main = pd.read_csv(filepath)
        print(f"Processing main file: {filename}")
    except Exception as e:
        print(f"Error loading main file '{filename}': {e}")
        continue

    # Load the allocations CSV file
    try:
        df_alloc = pd.read_csv(alloc_filepath)
        print(f"Processing allocations file: {alloc_filename}")
    except Exception as e:
        print(f"Error loading allocations file '{alloc_filename}': {e}")
        continue

    # Verify the presence of 'time_instant' column in main file
    if 'time_instant' not in df_main.columns:
        print(f"Error: 'time_instant' column not found in the main file '{filename}'. Skipping.")
        continue

    # Check for necessary columns in allocations file
    required_alloc_columns = ['job_id', 'submit_time', 'complete_time', 'num_gpu']
    if not all(col in df_alloc.columns for col in required_alloc_columns):
        print(f"Error: One or more required columns missing in allocations file '{alloc_filename}'. Skipping.")
        continue

    # Prepare job tracking DataFrame
    jobs_df = df_alloc[['job_id', 'submit_time', 'complete_time', 'num_gpu']].drop_duplicates().copy()

    # Ensure submit_time and complete_time are numeric using .loc to avoid SettingWithCopyWarning
    jobs_df.loc[:, 'submit_time'] = pd.to_numeric(jobs_df['submit_time'], errors='coerce')
    jobs_df.loc[:, 'complete_time'] = pd.to_numeric(jobs_df['complete_time'], errors='coerce')

    # Compute total GPU requested per job
    gpu_usage_per_job = df_alloc.groupby('job_id')['num_gpu'].sum().reset_index()
    gpu_usage_per_job.rename(columns={'num_gpu': 'total_gpu_requested'}, inplace=True)

    # Merge total_gpu_requested into jobs_df and make a copy to avoid the SettingWithCopyWarning
    jobs_df = jobs_df.merge(gpu_usage_per_job, on='job_id', how='left').copy()

    # Remove jobs with total_gpu_requested being NaN
    jobs_df = jobs_df.dropna(subset=['total_gpu_requested']).copy()

    # Determine quartiles
    quartile_edges = jobs_df['total_gpu_requested'].quantile([0.25, 0.5, 0.75]).values

    # Assign quartile to each job
    def assign_quartile(x, quartiles):
        if x <= quartiles[0]:
            return 'Q1'
        elif x <= quartiles[1]:
            return 'Q2'
        elif x <= quartiles[2]:
            return 'Q3'
        else:
            return 'Q4'
    jobs_df['Quartile'] = jobs_df['total_gpu_requested'].apply(lambda x: assign_quartile(x, quartile_edges))

    # Initialize lists to store metrics for this file
    jains_gpu = []
    ginis_gpu = []
    avg_gpu_util = []

    # Initialize dictionaries to store metrics per quartile
    quartiles_list = ['Q1', 'Q2', 'Q3', 'Q4']
    throughput_list = {q: [] for q in quartiles_list}
    job_latency_list = {q: [] for q in quartiles_list}
    queue_length_list = {q: [] for q in quartiles_list}
    active_jobs_list = {q: [] for q in quartiles_list}

    # Sort time_instants to ensure chronological order
    time_instants = sorted(df_main['time_instant'].unique())

    # ------------------------------
    # 4.1. Define the Time Range for Analysis
    # ------------------------------

    # Define the time range for analysis
    start_time = 0
    end_time = 1000  # Adjusted from 500 to 1000

    # Filter time instants within the specified range
    time_instants = [t for t in time_instants if start_time <= t <= end_time]

    # Create a DataFrame to store GPU utilizations
    gpu_util_df = pd.DataFrame(index=time_instants, columns=[f'Node_{i}' for i in range(100)])

    for t in time_instants:
        df_time = df_main[df_main['time_instant'] == t]

        # Handle multiple rows per time_instant by averaging numeric columns
        if df_time.shape[0] > 1:
            print(f"Multiple rows found for time_instant {t} in '{filename}'. Aggregating by averaging.")
            # Separate numeric and non-numeric columns
            numeric_cols = df_time.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude 'time_instant' from numeric_cols to prevent duplication
            numeric_cols = [col for col in numeric_cols if col != 'time_instant']
            non_numeric_cols = df_time.select_dtypes(exclude=[np.number]).columns.tolist()
            # Exclude 'time_instant' from non-numeric columns if present
            if 'time_instant' in non_numeric_cols:
                non_numeric_cols.remove('time_instant')
            # Group and mean numeric columns
            if numeric_cols:
                df_time_numeric = df_time.groupby('time_instant')[numeric_cols].mean().reset_index()
            else:
                # If no numeric columns, create an empty DataFrame with 'time_instant'
                df_time_numeric = df_time.groupby('time_instant').size().reset_index(name='count')
            # For non-numeric columns, take the first occurrence
            if non_numeric_cols:
                df_time_non_numeric = df_time.groupby('time_instant')[non_numeric_cols].first().reset_index()
                # Merge numeric and non-numeric DataFrames
                df_time = pd.merge(df_time_numeric, df_time_non_numeric, on='time_instant')
            else:
                df_time = df_time_numeric

        # Proceed with single-row processing
        if df_time.shape[0] != 1:
            print(f"Unexpected number of rows after aggregation for time_instant {t} in '{filename}'. Skipping this time instant.")
            continue

        gpu_util_list = []

        for node in range(100):
            node_prefix = f'node_{node}'
            initial_gpu_col = f'{node_prefix}_initial_gpu'
            used_gpu_col = f'{node_prefix}_used_gpu'

            # Check if required columns exist
            if initial_gpu_col not in df_time.columns or used_gpu_col not in df_time.columns:
                print(f"Missing GPU utilization columns for {node_prefix} in '{filename}'. Skipping this node.")
                gpu_util = 0
            else:
                initial_gpu = df_time.iloc[0][initial_gpu_col]
                used_gpu = df_time.iloc[0][used_gpu_col]
                # Handle potential non-numeric values
                try:
                    gpu_util = float(used_gpu) / float(initial_gpu) if float(initial_gpu) != 0 else 0
                except ValueError:
                    print(f"Non-numeric GPU utilization values for {node_prefix} at time_instant {t} in '{filename}'. Setting GPU utilization to 0.")
                    gpu_util = 0
            gpu_util_df.loc[t, f'Node_{node}'] = gpu_util
            gpu_util_list.append(gpu_util)

        # Compute GPU utilization metrics
        if gpu_util_list:
            jains_gpu.append(compute_jains_index(gpu_util_list))
            ginis_gpu.append(compute_ginis_index(gpu_util_list))
            avg_gpu_util.append(np.mean(gpu_util_list))
        else:
            jains_gpu.append(np.nan)
            ginis_gpu.append(np.nan)
            avg_gpu_util.append(np.nan)

        # Compute workload metrics per quartile
        for quartile in quartiles_list:
            jobs_in_quartile = jobs_df[jobs_df['Quartile'] == quartile]
            throughput = compute_throughput(jobs_in_quartile, t)
            job_lat = compute_job_latency(jobs_in_quartile, t)
            queue_len = compute_queue_length(jobs_in_quartile, t)
            active_jobs = compute_active_jobs(jobs_in_quartile, t)

            throughput_list[quartile].append(throughput)
            job_latency_list[quartile].append(job_lat)
            queue_length_list[quartile].append(queue_len)
            active_jobs_list[quartile].append(active_jobs)

    # Append overall node-based metrics
    master_metrics.append({
        'Utility_Function': utility_function,
        'Replication': rep,
        'Quartile': 'Overall',
        'Jains_GPU_Fairness': np.mean(jains_gpu) if jains_gpu else np.nan,
        'Ginis_GPU_Index': np.mean(ginis_gpu) if ginis_gpu else np.nan,
        'Average_GPU_Utilization': np.mean(avg_gpu_util) if avg_gpu_util else np.nan,
        'Throughput_Jobs_Completed': np.nan,
        'Average_Job_Latency': np.nan,
        'Queue_Length': np.nan,
        'Active_Jobs': np.nan
    })

    # Append job-based metrics per quartile
    for quartile in quartiles_list:
        master_metrics.append({
            'Utility_Function': utility_function,
            'Replication': rep,
            'Quartile': quartile,
            'Jains_GPU_Fairness': np.nan,
            'Ginis_GPU_Index': np.nan,
            'Average_GPU_Utilization': np.nan,
            'Throughput_Jobs_Completed': np.mean(throughput_list[quartile]) if throughput_list[quartile] else np.nan,
            'Average_Job_Latency': np.mean(job_latency_list[quartile]) if job_latency_list[quartile] else np.nan,
            'Queue_Length': np.mean(queue_length_list[quartile]) if queue_length_list[quartile] else np.nan,
            'Active_Jobs': np.mean(active_jobs_list[quartile]) if active_jobs_list[quartile] else np.nan
        })

# ------------------------------
# 5. Create Master DataFrame
# ------------------------------

master_df = pd.DataFrame(master_metrics)
print("Master DataFrame created.")

# Save the aggregated metrics to a CSV file
master_csv_path = os.path.join(plots_directory, 'aggregated_metrics.csv')
master_df.to_csv(master_csv_path, index=False)
print(f"Aggregated metrics saved to '{master_csv_path}'.")

# ------------------------------
# 6. Comparative Analysis and Visualization
# ------------------------------

# Define a color palette for different utility functions
utility_functions = master_df['Utility_Function'].unique()
quartiles_list = ['Q1', 'Q2', 'Q3', 'Q4']
palette = sns.color_palette("husl", len(utility_functions))

# 6.1. Compare Average GPU Utilization Across Utility Functions
node_based_df = master_df[master_df['Quartile'] == 'Overall']
plt.figure(figsize=(10, 6))
sns.boxplot(x='Utility_Function', y='Average_GPU_Utilization', data=node_based_df, palette=palette)
plt.xlabel('Utility Function')
plt.ylabel('Average GPU Utilization')
plt.title(f'Average GPU Utilization Across Utility Functions (Time {start_time} to {end_time})')
plt.grid(True, axis='y')
plt.tight_layout()
boxplot_path = os.path.join(plots_directory, 'average_gpu_utilization_comparison.png')
plt.savefig(boxplot_path)
plt.close()
print(f"Plot 'average_gpu_utilization_comparison.png' saved successfully.")

# 6.2. Compare Jain's GPU Fairness Index Across Utility Functions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Utility_Function', y='Jains_GPU_Fairness', data=node_based_df, palette=palette)
plt.xlabel('Utility Function')
plt.ylabel("Jain's GPU Fairness Index")
plt.title(f"Jain's GPU Fairness Index Across Utility Functions (Time {start_time} to {end_time})")
plt.grid(True, axis='y')
plt.tight_layout()
jains_boxplot_path = os.path.join(plots_directory, "jains_gpu_fairness_comparison.png")
plt.savefig(jains_boxplot_path)
plt.close()
print(f"Plot 'jains_gpu_fairness_comparison.png' saved successfully.")

# 6.3. Compare Gini's GPU Index Across Utility Functions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Utility_Function', y='Ginis_GPU_Index', data=node_based_df, palette=palette)
plt.xlabel('Utility Function')
plt.ylabel("Gini's GPU Index")
plt.title(f"Gini's GPU Index Across Utility Functions (Time {start_time} to {end_time})")
plt.grid(True, axis='y')
plt.tight_layout()
ginis_boxplot_path = os.path.join(plots_directory, "ginis_gpu_index_comparison.png")
plt.savefig(ginis_boxplot_path)
plt.close()
print(f"Plot 'ginis_gpu_index_comparison.png' saved successfully.")

# 6.4. Compare Throughput Across Utility Functions by Quartile
plt.figure(figsize=(12, 7))
sns.boxplot(x='Utility_Function', y='Throughput_Jobs_Completed', hue='Quartile', data=master_df[master_df['Quartile'] != 'Overall'])
plt.xlabel('Utility Function')
plt.ylabel('Throughput (Jobs Completed)')
plt.title(f'Throughput Across Utility Functions by Quartile (Time {start_time} to {end_time})')
plt.grid(True, axis='y')
plt.legend(title='Quartile')
plt.tight_layout()
throughput_boxplot_path = os.path.join(plots_directory, 'throughput_comparison_by_quartile.png')
plt.savefig(throughput_boxplot_path)
plt.close()
print(f"Plot 'throughput_comparison_by_quartile.png' saved successfully.")

# 6.5. Compare Average Job Latency Across Utility Functions by Quartile
plt.figure(figsize=(12, 7))
sns.boxplot(x='Utility_Function', y='Average_Job_Latency', hue='Quartile', data=master_df[master_df['Quartile'] != 'Overall'])
plt.xlabel('Utility Function')
plt.ylabel('Average Job Latency')
plt.title(f'Average Job Latency Across Utility Functions by Quartile (Time {start_time} to {end_time})')
plt.grid(True, axis='y')
plt.legend(title='Quartile')
plt.tight_layout()
job_latency_boxplot_path = os.path.join(plots_directory, 'job_latency_comparison_by_quartile.png')
plt.savefig(job_latency_boxplot_path)
plt.close()
print(f"Plot 'job_latency_comparison_by_quartile.png' saved successfully.")

# 6.6. Compare Queue Length Across Utility Functions by Quartile
plt.figure(figsize=(12, 7))
sns.boxplot(x='Utility_Function', y='Queue_Length', hue='Quartile', data=master_df[master_df['Quartile'] != 'Overall'])
plt.xlabel('Utility Function')
plt.ylabel('Queue Length')
plt.title(f'Queue Length Across Utility Functions by Quartile (Time {start_time} to {end_time})')
plt.grid(True, axis='y')
plt.legend(title='Quartile')
plt.tight_layout()
queue_length_boxplot_path = os.path.join(plots_directory, 'queue_length_comparison_by_quartile.png')
plt.savefig(queue_length_boxplot_path)
plt.close()
print(f"Plot 'queue_length_comparison_by_quartile.png' saved successfully.")

# 6.7. Compare Active Jobs Across Utility Functions by Quartile
plt.figure(figsize=(12, 7))
sns.boxplot(x='Utility_Function', y='Active_Jobs', hue='Quartile', data=master_df[master_df['Quartile'] != 'Overall'])
plt.xlabel('Utility Function')
plt.ylabel('Active Jobs')
plt.title(f'Active Jobs Across Utility Functions by Quartile (Time {start_time} to {end_time})')
plt.grid(True, axis='y')
plt.legend(title='Quartile')
plt.tight_layout()
active_jobs_boxplot_path = os.path.join(plots_directory, 'active_jobs_comparison_by_quartile.png')
plt.savefig(active_jobs_boxplot_path)
plt.close()
print(f"Plot 'active_jobs_comparison_by_quartile.png' saved successfully.")

# 6.8. Statistical Analysis (ANOVA)
# Example: Compare Average Job Latency across utility functions for each quartile
for quartile in quartiles_list:
    print(f"\nANOVA Results for Average Job Latency in Quartile {quartile}:")
    anova_df = master_df[(master_df['Quartile'] == quartile)][['Utility_Function', 'Average_Job_Latency']].dropna()
    groups = [group['Average_Job_Latency'].values for name, group in anova_df.groupby('Utility_Function')]

    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        f_val, p_val = stats.f_oneway(*groups)
        print(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")

        if p_val < 0.05:
            print("Significant differences found between utility functions.")
        else:
            print("No significant differences found between utility functions.")
    else:
        print("Not enough data for ANOVA.")

# Similarly, perform ANOVA for Throughput, Queue Length, and Active Jobs per quartile if needed

# ------------------------------
# 7. Summary of Results
# ------------------------------

# Display the master DataFrame
print("\nAggregated Metrics Across All Experiments:")
print(master_df)

# Save the master DataFrame to a CSV file
master_df.to_csv(os.path.join(plots_directory, 'aggregated_metrics_summary.csv'), index=False)
print(f"Aggregated metrics summary saved to '{os.path.join(plots_directory, 'aggregated_metrics_summary.csv')}'.")
