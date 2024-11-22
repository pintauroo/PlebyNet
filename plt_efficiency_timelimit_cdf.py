import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ------------------------------
# 1. Setup and Configuration
# ------------------------------

# Define the directory containing your CSV files
data_directory = '.'  # Adjust the path if necessary

# Define the directory to save plots
plots_directory = 'plots'
os.makedirs(plots_directory, exist_ok=True)

# Define the filename pattern for main files and allocations files
# Combined all regex patterns into one using alternation
main_file_regex = re.compile(
    r'(\d+)_70J_50N_NFD_HN_NDJ_(?:SPS_BW|SPS_NBW|MPS_BW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv'
)
alloc_file_suffix = '_jobs_report.csv'

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
selected_reps = range(1, 50)  # Replications 1 to 49

# Define the number of intervals
num_intervals = 2  # Change this value to 1, 2, 4, etc., as needed

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

def standardize_columns(df, required_columns):
    """
    Attempt to standardize column names in df to match required_columns.
    Returns a renamed DataFrame if possible, else returns None.
    """
    # Create a mapping from lower-case stripped names to actual column names
    df_columns = {col.lower().replace('_', '').replace(' ', ''): col for col in df.columns}
    mapping = {}
    for req_col in required_columns:
        req_col_std = req_col.lower().replace('_', '').replace(' ', '')
        if req_col_std in df_columns:
            mapping[req_col] = df_columns[req_col_std]
        else:
            # Try to find a close match
            possible_matches = [col for key, col in df_columns.items() if req_col_std in key or key in req_col_std]
            if possible_matches:
                mapping[req_col] = possible_matches[0]
            else:
                mapping[req_col] = None
    # Check if all required columns can be mapped
    if all(mapping[col] is not None for col in required_columns):
        # Rename columns to expected names
        rename_dict = {mapping[col]: col for col in required_columns}
        df = df.rename(columns=rename_dict)
        return df
    else:
        # Cannot map all required columns
        missing = [col for col in required_columns if mapping[col] is None]
        print(f"Missing required columns: {missing}")
        return None

# ------------------------------
# 3. Initialize Master DataFrame and GPU Request Dictionary
# ------------------------------

# Initialize a list to store all metrics
master_metrics = []

# Initialize a dictionary to store gpu_request_tot per file
gpu_request_tot_dict = {}

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

    # Load the allocations CSV file with diagnostic
    try:
        df_alloc = pd.read_csv(alloc_filepath)
        print(f"Processing allocations file: {alloc_filename}")
        print("Columns in df_alloc:", df_alloc.columns.tolist())  # Print columns to debug
    except Exception as e:
        print(f"Error loading allocations file '{alloc_filename}': {e}")
        continue

    # Adjust column names if necessary
    required_columns = ['job_id', 'submit_time', 'complete_time', 'num_gpu']
    df_alloc_standard = standardize_columns(df_alloc, required_columns)

    if df_alloc_standard is None:
        print(f"Skipping allocations file '{alloc_filename}' due to missing required columns.")
        continue

    # Proceed with standardized column names
    jobs_df = df_alloc_standard[['job_id', 'submit_time', 'complete_time', 'num_gpu']].drop_duplicates().copy()

    # Ensure submit_time and complete_time are numeric
    jobs_df['submit_time'] = pd.to_numeric(jobs_df['submit_time'], errors='coerce')
    jobs_df['complete_time'] = pd.to_numeric(jobs_df['complete_time'], errors='coerce')

    # ------------------------------
    # 4.a. Compute gpu_request_tot per file
    # ------------------------------
    gpu_request_tot = df_alloc_standard['num_gpu'].sum()
    gpu_request_tot_dict[(utility_function, rep)] = gpu_request_tot

    # ------------------------------
    # 4.b. Compute Additional Metrics
    # ------------------------------
    # Compute utilization ratios (Assuming 'num_gpu' represents utilization)
    # This is a placeholder; adjust based on actual data representation
    utilization_ratios = jobs_df['num_gpu']

    # Compute Jain's Fairness Index
    jains_index = compute_jains_index(utilization_ratios)

    # Compute Gini's Index
    ginis_index = compute_ginis_index(utilization_ratios)

    # Compute Throughput (Total jobs completed)
    # Assuming 'complete_time' represents time units, take the max as current_time
    current_time = jobs_df['complete_time'].max()
    throughput = compute_throughput(jobs_df, current_time)

    # Compute Job Latency (Average latency)
    job_latency = compute_job_latency(jobs_df, current_time)

    # Compute Queue Length at current_time
    queue_length = compute_queue_length(jobs_df, current_time)

    # Compute Active Jobs at current_time
    active_jobs = compute_active_jobs(jobs_df, current_time)

    # Append all metrics to master_metrics
    master_metrics.append({
        'Utility_Function': utility_function,
        'Replication': rep,
        'gpu_request_tot': gpu_request_tot,
        'jains_index': jains_index,
        'ginis_index': ginis_index,
        'throughput': throughput,
        'job_latency': job_latency,
        'queue_length': queue_length,
        'active_jobs': active_jobs
    })

# ------------------------------
# 5. Assign Intervals Based on gpu_request_tot
# ------------------------------

# Convert the gpu_request_tot_dict to a DataFrame
gpu_request_df = pd.DataFrame([
    {'Utility_Function': key[0], 'Replication': key[1], 'gpu_request_tot': value}
    for key, value in gpu_request_tot_dict.items()
])

# Compute quantile edges based on the number of intervals
quantiles = np.linspace(0, 1, num_intervals + 1)
quartile_edges = gpu_request_df['gpu_request_tot'].quantile(quantiles).values

# Define interval labels dynamically
interval_labels = [f"Q{i+1}" for i in range(num_intervals)]
interval_range_labels = []

for i in range(num_intervals):
    lower = quartile_edges[i]
    upper = quartile_edges[i + 1]
    interval_range_labels.append(f"Q{i+1} ({lower:.1f} - {upper:.1f} GPUs)")

# Function to assign interval
def assign_interval(x, edges):
    for i in range(len(edges) - 1):
        if i < len(edges) - 2:
            if edges[i] <= x < edges[i + 1]:
                return f"Q{i+1}"
        else:
            # Include the right edge in the last interval
            if edges[i] <= x <= edges[i + 1]:
                return f"Q{i+1}"
    return None  # In case x doesn't fall into any interval

# Assign intervals
gpu_request_df['Interval'] = gpu_request_df['gpu_request_tot'].apply(lambda x: assign_interval(x, quartile_edges))

# Handle any missing assignments
missing_intervals = gpu_request_df['Interval'].isnull().sum()
if missing_intervals > 0:
    print(f"Warning: {missing_intervals} entries did not fit into any interval.")

# Create a mapping for interval labels with ranges
interval_mapping = {f"Q{i+1}": interval_range_labels[i] for i in range(num_intervals)}
gpu_request_df['Interval_Range'] = gpu_request_df['Interval'].map(interval_mapping)

# ------------------------------
# 6. Save the Results in a Dictionary
# ------------------------------

# Convert gpu_request_df to a dictionary if needed
# For example, you can have interval-wise lists
interval_dict = {f"Q{i+1}": gpu_request_df[gpu_request_df['Interval'] == f"Q{i+1}"]['gpu_request_tot'].tolist() for i in range(num_intervals)}

# ------------------------------
# 7. Comparative Analysis and Visualization
# ------------------------------

# Define a color palette for different intervals
palette = sns.color_palette("husl", num_intervals)

# Plot CDF for gpu_request_tot divided into intervals
def plot_cdf_by_interval(interval_data, interval_labels, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for interval, label in zip(interval_data.keys(), interval_labels):
        values = interval_data.get(interval, [])
        if not values:
            print(f"No data for {interval}; skipping CDF plot for this interval.")
            continue
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Plot '{filename}' saved successfully.")

# Prepare data for plotting
plot_data = interval_dict
plot_labels = interval_range_labels

# Plot CDF for gpu_request_tot
plot_cdf_by_interval(
    interval_data=plot_data,
    interval_labels=plot_labels,
    xlabel='Total GPU Requested',
    ylabel='CDF',
    title='CDF of Total GPU Requested by Intervals',
    filename=os.path.join(plots_directory, f"gpu_request_tot_cdf_by_intervals_{num_intervals}.png")
)

# ------------------------------
# 7.a. Plot Additional Metrics
# ------------------------------

# Convert master_metrics to DataFrame
metrics_df = pd.DataFrame(master_metrics)

# Merge interval information
metrics_df = metrics_df.merge(
    gpu_request_df[['Utility_Function', 'Replication', 'Interval']],
    on=['Utility_Function', 'Replication'],
    how='left'
)

# Set the theme for seaborn
sns.set_theme(style="whitegrid")

# List of metrics to plot
metrics_to_plot = ['gpu_request_tot', 'jains_index', 'ginis_index', 'throughput', 'job_latency', 'queue_length', 'active_jobs']

# Function to create boxplots for each metric
def plot_boxplots(df, metric, plots_dir, interval_labels):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Utility_Function', y=metric, hue='Interval', data=df, palette=palette)
    plt.title(f'Boxplot of {metric.replace("_", " ").title()} by Utility Function and Interval')
    plt.xlabel('Utility Function')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend(title='Interval')
    plt.tight_layout()
    filename = os.path.join(plots_dir, f"{metric}_boxplot_{num_intervals}_intervals.png")
    plt.savefig(filename)
    plt.close()
    print(f"Boxplot for '{metric}' saved as '{filename}'.")

# Function to create distribution plots (e.g., histograms) for each metric
def plot_distributions(df, metric, plots_dir, interval_labels):
    plt.figure(figsize=(10, 6))
    try:
        # Check if data has enough unique values for KDE
        unique_values = df[metric].nunique()
        if unique_values < 2:
            raise ValueError("Not enough unique values for KDE.")

        # Additionally, check if the data is not all identical
        if df[metric].nunique() == 1:
            raise ValueError("All data points have the same value; KDE cannot be computed.")

        # Plot with KDE
        sns.histplot(data=df, x=metric, hue='Interval', multiple='stack', palette=palette, kde=True)
    except Exception as e:
        print(f"Could not plot KDE for '{metric}': {e}")
        # Plot without KDE
        sns.histplot(data=df, x=metric, hue='Interval', multiple='stack', palette=palette, kde=False)

    plt.title(f'Distribution of {metric.replace("_", " ").title()} by Interval')
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel('Count')
    plt.tight_layout()
    filename = os.path.join(plots_dir, f"{metric}_distribution_{num_intervals}_intervals.png")
    plt.savefig(filename)
    plt.close()
    print(f"Distribution plot for '{metric}' saved as '{filename}'.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# Plot each metric
for metric in metrics_to_plot:
    plot_boxplots(metrics_df, metric, plots_directory, interval_range_labels)
    plot_distributions(metrics_df, metric, plots_directory, interval_range_labels)

# ------------------------------
# 8. Save the Aggregated Metrics
# ------------------------------

# Merge interval range information back to master_metrics if needed
# This step is optional and depends on how you want to use master_metrics
master_metrics_df = pd.DataFrame(master_metrics)
master_metrics_df = master_metrics_df.merge(
    gpu_request_df[['Utility_Function', 'Replication', 'Interval_Range']],
    on=['Utility_Function', 'Replication'],
    how='left'
)

# Save the aggregated metrics to a CSV file
master_csv_path = os.path.join(plots_directory, 'aggregated_metrics.csv')
master_metrics_df.to_csv(master_csv_path, index=False)
print(f"Aggregated metrics saved to '{master_csv_path}'.")

# ------------------------------
# 9. Summary of Results
# ------------------------------

# Display the gpu_request_tot with intervals
print("\nTotal GPU Requests per File with Intervals:")
print(gpu_request_df)

# Save the gpu_request_df to a CSV file
gpu_request_csv_path = os.path.join(plots_directory, f'gpu_request_tot_intervals_{num_intervals}.csv')
gpu_request_df.to_csv(gpu_request_csv_path, index=False)
print(f"GPU request totals with intervals saved to '{gpu_request_csv_path}'.")
