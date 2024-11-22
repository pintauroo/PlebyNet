import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Setup and Configuration
# ------------------------------

# Define the directory containing your CSV files
data_directory = '.'  # Adjust the path if necessary

# Define the base directory to save plots and aggregated data
base_plots_directory = 'plots'
os.makedirs(base_plots_directory, exist_ok=True)

# Define the number of intervals (quartiles by default)
num_intervals = 4  # You can change this value to set different interval counts

# Define a list of configurations, each with its own regex pattern and plot configurations
file_configurations = [
    {
        'name': 'SPS_BW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_SPS_BW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        ),
        'alloc_suffix': '_jobs_report.csv',
        'plot_subdir': 'SPS_BW_plots'  # Subdirectory for plots related to SPS_BW
    },
    {
        'name': 'SPS_NBW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_SPS_NBW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        ),
        'alloc_suffix': '_jobs_report.csv',
        'plot_subdir': 'SPS_NBW_plots'  # Subdirectory for plots related to SPS_NBW
    },
    {
        'name': 'MPS_BW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_MPS_BW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        ),
        'alloc_suffix': '_jobs_report.csv',
        'plot_subdir': 'MPS_BW_plots'  # Subdirectory for plots related to MPS_BW
    },
]

# Define the list of utilities
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']

# Define the range of replications
selected_reps = range(1, 50)  # Replications 1 to 49

# ------------------------------
# 2. Define Utility Functions
# ------------------------------

def standardize_columns(df, required_columns):
    """
    Attempt to standardize column names in df to match required_columns.
    Returns a renamed DataFrame if possible, else returns None.
    """
    # Create a mapping from lower-case stripped names to actual column names
    df_columns = {col.lower().replace('_', '').replace(' ', ''): col
                 for col in df.columns}
    mapping = {}
    for req_col in required_columns:
        req_col_std = req_col.lower().replace('_', '').replace(' ', '')
        if req_col_std in df_columns:
            mapping[req_col] = df_columns[req_col_std]
        else:
            # Try to find a close match
            possible_matches = [
                col for key, col in df_columns.items()
                if req_col_std in key or key in req_col_std
            ]
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

def create_plots_directory(base_dir, subdir):
    """
    Create a subdirectory for plots if it doesn't exist.
    """
    plots_dir = os.path.join(base_dir, subdir)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

# ------------------------------
# 3. Initialize Data Structures
# ------------------------------

# Initialize a list to store all job metrics across all configurations
all_jobs_metrics = []

# Initialize a list to store JCT statistics across all configurations
all_jct_stats = []

# ------------------------------
# 4. Process Each Configuration and Its Files
# ------------------------------

for config in file_configurations:
    config_name = config['name']
    config_pattern = config['pattern']
    alloc_suffix = config['alloc_suffix']
    plots_subdir = config['plot_subdir']
    
    # Create a subdirectory for plots related to this configuration
    plots_directory = create_plots_directory(base_plots_directory, plots_subdir)
    
    print(f"\nProcessing Configuration: {config_name}")
    
    # List all main CSV files matching the current configuration pattern
    config_file_list = [f for f in os.listdir(data_directory)
                        if config_pattern.match(f)]
    
    if not config_file_list:
        print(f"  - No main CSV files found for configuration '{config_name}'. Skipping.")
        continue
    
    print(f"  - Found {len(config_file_list)} main files for '{config_name}'.")
    
    for filename in config_file_list:
        match = config_pattern.match(filename)
        if not match:
            print(f"    - Filename '{filename}' does not match the expected pattern. Skipping.")
            continue
        replication_number = match.group(1)
        utility_function = match.group(2)
    
        # Determine the corresponding allocations file
        alloc_filename = filename.replace('.csv', alloc_suffix)
        alloc_filepath = os.path.join(data_directory, alloc_filename)
    
        if not os.path.exists(alloc_filepath):
            print(f"    - Allocations file '{alloc_filename}' not found for '{filename}'. Skipping.")
            continue
    
        # Load the allocations CSV file
        try:
            df_alloc = pd.read_csv(alloc_filepath)
            print(f"    - Processing allocations file: {alloc_filename}")
        except Exception as e:
            print(f"    - Error loading allocations file '{alloc_filename}': {e}")
            continue
    
        # Define required columns, including 'duration'
        required_columns = ['job_id', 'submit_time',
                            'complete_time', 'num_gpu', 'duration']
        df_alloc_standard = standardize_columns(df_alloc, required_columns)
    
        if df_alloc_standard is None:
            print(
                f"    - Skipping allocations file '{alloc_filename}' due to missing required columns.")
            continue
    
        # Proceed with standardized column names
        jobs_df = df_alloc_standard[['job_id', 'submit_time',
                                     'complete_time', 'num_gpu', 'duration']].drop_duplicates().copy()
    
        # Ensure numeric columns are of numeric type
        jobs_df['submit_time'] = pd.to_numeric(
            jobs_df['submit_time'], errors='coerce')
        jobs_df['complete_time'] = pd.to_numeric(
            jobs_df['complete_time'], errors='coerce')
        jobs_df['num_gpu'] = pd.to_numeric(jobs_df['num_gpu'], errors='coerce')
        jobs_df['duration'] = pd.to_numeric(jobs_df['duration'], errors='coerce')
    
        # Remove jobs with missing data
        jobs_df = jobs_df.dropna(subset=[
                                 'job_id', 'submit_time', 'complete_time', 'num_gpu', 'duration'])
    
        # Compute Job Completion Time (JCT)
        jobs_df['JCT'] = jobs_df['complete_time'] - jobs_df['submit_time']
    
        # Filter out jobs with non-positive JCT or non-positive duration
        jobs_df = jobs_df[(jobs_df['JCT'] >= 0) & (jobs_df['duration'] > 0)]
    
        # Compute Scaled JCT
        jobs_df['JCT_Scaled'] = jobs_df['JCT'] / jobs_df['duration']
    
        # Debugging: Check Scaled JCT ranges
        print(f"    - Utility: {utility_function}")
        print(f"      - Number of jobs after scaling: {len(jobs_df)}")
        print(f"      - Scaled JCT stats:\n{jobs_df['JCT_Scaled'].describe()}\n")
    
        # Append Configuration and Utility Function info
        jobs_df['Utility_Function'] = utility_function
        jobs_df['Configuration'] = config_name
        jobs_df['Replication'] = int(replication_number)
    
        # Collect all job data for later analysis
        all_jobs_metrics.append(jobs_df)

# ------------------------------
# 5. Combine All Jobs Data
# ------------------------------

if not all_jobs_metrics:
    print("\nNo job data collected from any configuration. Exiting.")
    exit(1)

# Concatenate all job data into a single DataFrame
all_jobs_df = pd.concat(all_jobs_metrics, ignore_index=True)

print(f"\nTotal number of jobs collected across all configurations: {len(all_jobs_df)}")

# ------------------------------
# 6. Categorize Jobs into Intervals Based on GPU Requirement
# ------------------------------

# Assign intervals using pd.qcut with a dynamic number of intervals
try:
    all_jobs_df['Interval'], bins = pd.qcut(
        all_jobs_df['num_gpu'],
        q=num_intervals,
        labels=False,
        retbins=True,
        duplicates='drop'
    )
except ValueError as e:
    print(f"\nError in interval assignment: {e}")
    exit(1)

# Extract unique bin edges
unique_bins = np.unique(bins)

# Calculate the number of intervals assigned
num_actual_intervals = len(unique_bins) - 1  # number of intervals

print(f"\nNumber of intervals assigned: {num_actual_intervals}")

# Define interval labels based on bins
interval_labels = []
labels_available = []

for i in range(num_actual_intervals):
    interval_labels.append(
        f"I{i+1} ({unique_bins[i]:.1f} - {unique_bins[i+1]:.1f} GPUs)"
    )
    labels_available.append(f"I{i+1}")

# Create mapping from labels_available to interval_labels
interval_mapping = dict(zip(labels_available, interval_labels))

# Map intervals to labels
all_jobs_df['Interval'] = all_jobs_df['Interval'].astype(int).map(
    lambda x: labels_available[x] if x < num_actual_intervals else np.nan)

all_jobs_df['Interval_Range'] = all_jobs_df['Interval'].map(interval_mapping)

# Verify interval assignments
print("\nInterval Distribution:")
print(all_jobs_df['Interval_Range'].value_counts())

# ------------------------------
# 7. Analyze JCT Differences Between Intervals Across Utility Functions and Configurations
# ------------------------------

# Group by Configuration, Utility Function, and Interval, then compute aggregate statistics
jct_stats = all_jobs_df.groupby(
    ['Configuration', 'Utility_Function', 'Interval']).agg(
    JCT_Mean=('JCT', 'mean'),
    JCT_Median=('JCT', 'median'),
    JCT_STD=('JCT', 'std'),
    Job_Count=('JCT', 'count')
).reset_index()

# Merge interval range labels
jct_stats['Interval_Range'] = jct_stats['Interval'].map(interval_mapping)

# Handle cases where Interval_Range might be missing
jct_stats['Interval_Range'] = jct_stats['Interval_Range'].fillna(
    jct_stats['Interval'])

# Debugging: Check jct_stats
print("\nJCT Statistics by Configuration, Utility Function, and Interval:")
print(jct_stats)

# ------------------------------
# 8. Visualization
# ------------------------------

# Set the plotting style
sns.set(style="whitegrid")

def plot_jct_scaled_cdf(df, plots_dir, config_name):
    """
    Plot the CDF of scaled JCT (JCT_Scaled) for each utility function and configuration within each interval.
    All configurations and utilities are overlaid in the same plot for direct comparison.
    """
    intervals = df['Interval_Range'].unique()
    # Sort intervals based on the lower bound of GPU range
    intervals_sorted = sorted(intervals, key=lambda x: float(x.split('(')[1].split()[0]))

    for interval in intervals_sorted:
        plt.figure(figsize=(14, 8))
        subset_interval = df[df['Interval_Range'] == interval]
        if subset_interval.empty:
            print(f"\n - No data available for interval '{interval}'. Skipping plot.")
            plt.close()
            continue
        # Define hue as a combination of Configuration and Utility_Function
        subset_interval['Config_Utility'] = subset_interval['Configuration'] + ' - ' + subset_interval['Utility_Function']
        utility_configs = subset_interval['Config_Utility'].unique()

        # Debugging: Check utilities and configurations present
        print(f"\nPlotting CDF for {config_name} - {interval}:")
        print(f" - Combined Utilities present: {utility_configs.tolist()}")

        for cu in utility_configs:
            subset = subset_interval[subset_interval['Config_Utility'] == cu]
            if subset.empty:
                print(f"   - No data for '{cu}'. Skipping.")
                continue
            sorted_jct_scaled = np.sort(subset['JCT_Scaled'])
            cdf = np.arange(1, len(sorted_jct_scaled)+1) / len(sorted_jct_scaled)
            plt.plot(sorted_jct_scaled, cdf, label=cu)

        plt.title(f"CDF of Scaled JCT for {interval} ({config_name})", fontsize=16)
        plt.xlabel('Scaled Job Completion Time (JCT / Duration)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        plt.legend(title='Configuration - Utility', fontsize=10, title_fontsize=12, loc='lower right')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        # Create a safe filename by replacing spaces and parentheses
        safe_interval = interval.replace(' ', '_').replace('(', '').replace(')', '').replace('-', 'to')
        plot_filename = os.path.join(plots_dir, f'cdf_jct_scaled_{config_name}_{safe_interval}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f" - CDF plot saved as '{plot_filename}'.")

def plot_jct_scaled_survival(df, plots_dir, config_name):
    """
    Plot the Survival Function (1 - CDF) of scaled JCT (JCT_Scaled) for each utility function and configuration within each interval.
    All configurations and utilities are overlaid in the same plot for direct comparison.
    """
    intervals = df['Interval_Range'].unique()
    # Sort intervals based on the lower bound of GPU range
    intervals_sorted = sorted(intervals, key=lambda x: float(x.split('(')[1].split()[0]))

    for interval in intervals_sorted:
        plt.figure(figsize=(14, 8))
        subset_interval = df[df['Interval_Range'] == interval]
        if subset_interval.empty:
            print(f"\n - No data available for interval '{interval}'. Skipping survival plot.")
            plt.close()
            continue
        # Define hue as a combination of Configuration and Utility_Function
        subset_interval['Config_Utility'] = subset_interval['Configuration'] + ' - ' + subset_interval['Utility_Function']
        utility_configs = subset_interval['Config_Utility'].unique()

        # Debugging: Check utilities and configurations present
        print(f"\nPlotting Survival Function for {config_name} - {interval}:")
        print(f" - Combined Utilities present: {utility_configs.tolist()}")

        for cu in utility_configs:
            subset = subset_interval[subset_interval['Config_Utility'] == cu]
            if subset.empty:
                print(f"   - No data for '{cu}'. Skipping.")
                continue
            sorted_jct_scaled = np.sort(subset['JCT_Scaled'])
            cdf = np.arange(1, len(sorted_jct_scaled)+1) / len(sorted_jct_scaled)
            survival = 1 - cdf
            plt.plot(sorted_jct_scaled, survival, label=cu)

        plt.title(f"Survival Function of Scaled JCT for {interval} ({config_name})", fontsize=16)
        plt.xlabel('Scaled Job Completion Time (JCT / Duration)', fontsize=14)
        plt.ylabel('Survival Function (1 - CDF)', fontsize=14)
        plt.legend(title='Configuration - Utility', fontsize=10, title_fontsize=12, loc='upper right')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        # Create a safe filename by replacing spaces and parentheses
        safe_interval = interval.replace(' ', '_').replace('(', '').replace(')', '').replace('-', 'to')
        plot_filename = os.path.join(plots_dir, f'survival_jct_scaled_{config_name}_{safe_interval}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f" - Survival Function plot saved as '{plot_filename}'.")

def plot_num_gpu_distribution(df, plots_dir, config_name):
    """
    Plot the distribution of GPU requirements across all jobs.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='num_gpu', bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of GPU Requirements Across All Jobs ({config_name})', fontsize=16)
    plt.xlabel('Number of GPUs Requested', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = os.path.join(plots_dir, f'num_gpu_distribution_{config_name}.png')
    plt.savefig(filename)
    plt.close()
    print(f"\nHistogram of GPU distribution saved as '{filename}'.")

def plot_jct_boxplot(df, plots_dir, config_name):
    """
    Plot a boxplot of Job Completion Time (JCT) by Interval and Utility Function.
    """
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='Interval_Range', y='JCT', hue='Utility_Function', data=df)
    plt.title(f'Boxplot of Job Completion Time (JCT) by Interval and Utility Function ({config_name})', fontsize=16)
    plt.xlabel('GPU Requirement Intervals', fontsize=14)
    plt.ylabel('Job Completion Time (JCT)', fontsize=14)
    plt.legend(title='Utility Functions', fontsize=12, title_fontsize=14, loc='upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = os.path.join(plots_dir, f'jct_boxplot_{config_name}.png')
    plt.savefig(filename)
    plt.close()
    print(f"\nBoxplot of JCT saved as '{filename}'.")

# ------------------------------
# 9. Execute Visualization Functions
# ------------------------------

# Iterate over each configuration to generate plots
for config in file_configurations:
    config_name = config['name']
    plots_subdir = config['plot_subdir']
    plots_directory = os.path.join(base_plots_directory, plots_subdir)
    
    # Filter the combined DataFrame for the current configuration
    df_config = all_jobs_df[all_jobs_df['Configuration'] == config_name]
    
    if df_config.empty:
        print(f"\nNo data available for configuration '{config_name}'. Skipping plotting.")
        continue
    
    # Generate CDF Plots
    plot_jct_scaled_cdf(df_config, plots_directory, config_name)
    
    # Generate Survival Function Plots
    plot_jct_scaled_survival(df_config, plots_directory, config_name)
    
    # Generate GPU Distribution Histogram
    plot_num_gpu_distribution(df_config, plots_directory, config_name)
    
    # Generate JCT Boxplot
    plot_jct_boxplot(df_config, plots_directory, config_name)

# ------------------------------
# 10. Save the Aggregated Metrics
# ------------------------------

# Define a directory to save aggregated metrics
metrics_directory = os.path.join(base_plots_directory, 'aggregated_metrics')
os.makedirs(metrics_directory, exist_ok=True)

# Save the JCT statistics to a CSV file
jct_stats_csv_path = os.path.join(
    metrics_directory, 'jct_stats_by_configuration_utility_interval.csv')
jct_stats.to_csv(jct_stats_csv_path, index=False)
print(f"\nJCT statistics saved to '{jct_stats_csv_path}'.")

# Save all job data with interval assignments to a CSV file
all_jobs_csv_path = os.path.join(
    metrics_directory, 'all_jobs_with_intervals_and_configurations.csv')
all_jobs_df.to_csv(all_jobs_csv_path, index=False)
print(f"All job data with intervals and configurations saved to '{all_jobs_csv_path}'.")

# ------------------------------
# 11. Summary of Results
# ------------------------------

# Display the JCT statistics
print("\nJob Completion Time (JCT) Statistics by Configuration, Utility Function, and Interval:")
print(jct_stats)

# Optionally, display the first few rows of all_jobs_df
# print("\nFirst few rows of all_jobs_df:")
# print(all_jobs_df.head())
