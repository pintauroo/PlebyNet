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

# Define the base directory to save all plots
base_plots_directory = 'plots'
os.makedirs(base_plots_directory, exist_ok=True)

# Define file configurations
file_configurations = [
    {
        'name': 'SPS_BW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_SPS_BW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        ),
        'alloc_suffix': '_allocations.csv',
        'plot_subdir': 'SPS_BW_plots'
    },
    {
        'name': 'SPS_NBW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_SPS_NBW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        )import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import os

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def remove_outliers_iqr(df, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            initial_count = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            final_count = len(df)
            removed = initial_count - final_count
            if removed > 0:
                print(f"Removed {removed} outliers from '{col}'.")
    return df

def plot_confidence_intervals_by_utility(csv_file, label, confidence=0.95, t_gpu_min=None, t_gpu_max=None, output_dir='plots/confidence_intervals', text_size=14):
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file + '.csv')
    except FileNotFoundError:
        print(f"Error: File '{csv_file}.csv' not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: CSV file is malformed.")
        return

    # Apply t_gpu filters if specified
    if t_gpu_min is not None:
        df = df[df['t_gpu'] >= t_gpu_min]
    if t_gpu_max is not None:
        df = df[df['t_gpu'] < t_gpu_max]
    print(f"Number of rows after t_gpu filtering: {len(df)}")

    # Ensure 'utility' column exists
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return

    # Drop rows where 'utility' is NaN
    initial_row_count = len(df)
    df = df.dropna(subset=['utility'])
    final_row_count = len(df)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN in 'utility' column.")

    # Define the utility type mapping
    utility_mapping = {
        'Utility.UTIL': 'FRAG',
        'Utility.SGF': 'SGF',
        'Utility.LGF': 'LGF',
        'Utility.SEQ': 'SEQ',
        'Utility.LIKELIHOOD': 'LIKELIHOOD',
        'Utility.DRF': 'DRF',
        'Utility.TETRIS': 'TETRIS'
    }

    # Rename utility types
    df['utility'] = df['utility'].map(utility_mapping)

    # Check for any unmapped utility types
    unmapped_utilities = df['utility'].isna()
    if unmapped_utilities.any():
        unique_unmapped = df.loc[unmapped_utilities, 'utility'].unique()
        print(f"Warning: Found unmapped utility types: {unique_unmapped}")
        df = df.dropna(subset=['utility'])
        print(f"Dropped rows with unmapped utility types.")

    # Select the specified numerical columns
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned', 
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]

    # Verify that the selected columns exist in the dataframe
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing in the CSV file: {missing_columns}")
        return

    # Remove outliers
    df = remove_outliers_iqr(df, selected_columns)

    numeric_df = df[selected_columns].copy()

    # Combine the selected numerical columns with the 'utility' column
    numeric_df['utility'] = df['utility']

    # Get unique utility types after renaming
    utility_types = numeric_df['utility'].unique()
    print(f"Utility Types after renaming: {utility_types}")
    num_utilities = len(utility_types)

    # Compute mean and confidence intervals for each numerical column grouped by utility
    summary_data = {}
    for col in selected_columns:
        summary_data[col] = {}
        for utility in utility_types:
            group_data = numeric_df[numeric_df['utility'] == utility][col].dropna()
            if col == 'first_unassigned':
                group_data = group_data / 70  # Normalize by total number of jobs
            if group_data.empty:
                print(f"Warning: No valid data for utility '{utility}' in column '{col}'.")
                mean = np.nan
                ci = np.nan
            else:
                mean = group_data.mean()
                ci = compute_confidence_interval(group_data, confidence=confidence)
            summary_data[col][utility] = {'mean': mean, 'ci': ci}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_types_sorted = sorted(utility_types)
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types_sorted)}

    for col in selected_columns:
        # Create new figure for each plot
        plt.figure(figsize=(7, 6))
        means = [summary_data[col][utility]['mean'] for utility in utility_types_sorted]
        cis = [summary_data[col][utility]['ci'] for utility in utility_types_sorted]
        x_pos = np.arange(len(utility_t
,
        'alloc_suffix': '_allocations.csv',
        'plot_subdir': 'SPS_NBW_plots'
    },
    {
        'name': 'MPS_BW',
        'pattern': re.compile(
            r'^(\d+)_70J_50N_NFD_HN_NDJ_MPS_BW_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv$'
        ),
        'alloc_suffix': '_allocations.csv',
        'plot_subdir': 'MPS_BW_plots'
    },
]

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
selected_reps = range(1, 40)

# ------------------------------
# 2. Define Utility Functions
# ------------------------------

def compute_jains_index(x):
    x = np.array(x)
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x ** 2)
    n = len(x)
    return (sum_x ** 2) / (n * sum_x_sq) if sum_x_sq != 0 else 0

def compute_ginis_index(x):
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
    if jobs_df is None:
        return 0
    completed_jobs = jobs_df[jobs_df['completion_time'] == current_time]
    return len(completed_jobs)

def compute_job_latency(jobs_df, current_time):
    if jobs_df is None:
        return 0
    completed_jobs = jobs_df[jobs_df['completion_time'] == current_time]
    if completed_jobs.empty:
        return 0
    latencies = completed_jobs['completion_time'] - completed_jobs['arrival_time']
    return latencies.mean()

def compute_queue_length(jobs_df, current_time):
    if jobs_df is None:
        return 0
    pending_jobs = jobs_df[(jobs_df['arrival_time'] <= current_time) & (jobs_df['completion_time'] > current_time)]
    return len(pending_jobs)

def compute_active_jobs(jobs_df, current_time):
    if jobs_df is None:
        return 0
    active_jobs = jobs_df[(jobs_df['arrival_time'] <= current_time) & (jobs_df['completion_time'] > current_time)]
    return len(active_jobs)

# ------------------------------
# 3. Processing and Plotting Functions
# ------------------------------

def process_files_for_configuration(config):
    name = config['name']
    pattern = config['pattern']
    alloc_suffix = config['alloc_suffix']
    plot_subdir = config['plot_subdir']
    
    # Create plot subdirectory
    config_plots_dir = os.path.join(base_plots_directory, plot_subdir)
    os.makedirs(config_plots_dir, exist_ok=True)
    
    # Define the path for the aggregated metrics CSV
    aggregated_csv_path = os.path.join(config_plots_dir, 'aggregated_metrics.csv')
    
    # If aggregated CSV exists, load and return it to avoid reprocessing
    if os.path.exists(aggregated_csv_path):
        print(f"Aggregated metrics for '{name}' found. Loading from '{aggregated_csv_path}'.")
        return pd.read_csv(aggregated_csv_path)
    
    # List all main CSV files matching the pattern
    file_list = [f for f in os.listdir(data_directory) if pattern.match(f)]
    
    if not file_list:
        print(f"No main CSV files found matching the pattern for '{name}' in the specified directory.")
        return pd.DataFrame()  # Return empty DataFrame
    
    print(f"Found {len(file_list)} main files to process for configuration '{name}'.")
    
    # Initialize master metrics list
    master_metrics = []
    
    for filename in file_list:
        match = pattern.match(filename)
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
        alloc_filename = filename.replace('.csv', alloc_suffix)
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
        assert 'time_instant' in df_main.columns, f"'time_instant' column missing in '{filename}'."
        
        # Check for necessary columns in allocations file
        required_alloc_columns = ['job_id', 'submit_time', 'complete_time']
        missing_alloc_columns = [col for col in required_alloc_columns if col not in df_alloc.columns]
        assert not missing_alloc_columns, f"Missing columns {missing_alloc_columns} in allocations file '{alloc_filename}'."
        
        # Prepare job tracking DataFrame
        jobs_df = df_alloc[['job_id', 'submit_time', 'complete_time']].drop_duplicates()
        jobs_df.rename(columns={'submit_time': 'arrival_time', 'complete_time': 'completion_time'}, inplace=True)
        # Ensure arrival_time and completion_time are numeric
        jobs_df['arrival_time'] = pd.to_numeric(jobs_df['arrival_time'], errors='coerce')
        jobs_df['completion_time'] = pd.to_numeric(jobs_df['completion_time'], errors='coerce')
        
        # Initialize lists to store metrics for this file
        jains_gpu = []
        ginis_gpu = []
        avg_gpu_util = []
        throughput_list = []
        job_latency_list = []
        queue_length_list = []
        active_jobs_list = []
        
        # Sort time_instants to ensure chronological order
        time_instants = sorted(df_main['time_instant'].unique())
        
        # Iterate through each time instant
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
            
            for node in range(50):
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
                    except (ValueError, TypeError):
                        print(f"Non-numeric GPU utilization values for {node_prefix} at time_instant {t} in '{filename}'. Setting GPU utilization to 0.")
                        gpu_util = 0
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
            
            # Compute workload metrics
            throughput = compute_throughput(jobs_df, t)
            job_lat = compute_job_latency(jobs_df, t)
            queue_len = compute_queue_length(jobs_df, t)
            active_jobs = compute_active_jobs(jobs_df, t)
            
            throughput_list.append(throughput)
            job_latency_list.append(job_lat)
            queue_length_list.append(queue_len)
            active_jobs_list.append(active_jobs)
        
        # Append metrics for this experiment to the master list
        master_metrics.append({
            'Utility_Function': utility_function,
            'Replication': rep,
            'Jains_GPU_Fairness': np.mean(jains_gpu) if jains_gpu else np.nan,
            'Ginis_GPU_Index': np.mean(ginis_gpu) if ginis_gpu else np.nan,
            'Average_GPU_Utilization': np.mean(avg_gpu_util) if avg_gpu_util else np.nan,
            'Throughput_Jobs_Completed': np.mean(throughput_list) if throughput_list else np.nan,
            'Average_Job_Latency': np.mean(job_latency_list) if job_latency_list else np.nan,
            'Queue_Length': np.mean(queue_length_list) if queue_length_list else np.nan,
            'Active_Jobs': np.mean(active_jobs_list) if active_jobs_list else np.nan
        })
    
    # Create Master DataFrame
    master_df = pd.DataFrame(master_metrics)
    print(f"Master DataFrame created for configuration '{name}'.")
    
    # Save the aggregated metrics to a CSV file
    master_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated metrics saved to '{aggregated_csv_path}'.")
    
    return master_df

def plot_metrics(master_df, config, plot_subdir):
    name = config['name']
    config_plots_dir = os.path.join(base_plots_directory, plot_subdir)
    
    if master_df.empty:
        print(f"No data available for plotting in configuration '{name}'.")
        return
    
    # Define a color palette for different utility functions
    utility_functions = master_df['Utility_Function'].unique()
    palette = sns.color_palette("husl", len(utility_functions))
    
    # Define the plots to generate
    plots_info = [
        {
            'y': 'Average_GPU_Utilization',
            'ylabel': 'Average GPU Utilization',
            'title': 'Average GPU Utilization Across Utility Functions',
            'filename': 'average_gpu_utilization_comparison.png'
        },
        {
            'y': 'Jains_GPU_Fairness',
            'ylabel': "Jain's GPU Fairness Index",
            'title': "Jain's GPU Fairness Index Across Utility Functions",
            'filename': 'jains_gpu_fairness_comparison.png'
        },
        {
            'y': 'Ginis_GPU_Index',
            'ylabel': "Gini's GPU Index",
            'title': "Gini's GPU Index Across Utility Functions",
            'filename': 'ginis_gpu_index_comparison.png'
        },
        {
            'y': 'Throughput_Jobs_Completed',
            'ylabel': 'Throughput (Jobs Completed)',
            'title': 'Throughput Across Utility Functions',
            'filename': 'throughput_comparison.png'
        },
        {
            'y': 'Average_Job_Latency',
            'ylabel': 'Average Job Latency',
            'title': 'Average Job Latency Across Utility Functions',
            'filename': 'job_latency_comparison.png'
        },
        {
            'y': 'Queue_Length',
            'ylabel': 'Queue Length',
            'title': 'Queue Length Across Utility Functions',
            'filename': 'queue_length_comparison.png'
        },
        {
            'y': 'Active_Jobs',
            'ylabel': 'Active Jobs',
            'title': 'Active Jobs Across Utility Functions',
            'filename': 'active_jobs_comparison.png'
        },
    ]
    
    for plot in plots_info:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Utility_Function', y=plot['y'], data=master_df, palette=palette)
        plt.xlabel('Utility Function')
        plt.ylabel(plot['ylabel'])
        plt.title(plot['title'])
        plt.grid(True, axis='y')
        plt.tight_layout()
        plot_path = os.path.join(config_plots_dir, plot['filename'])
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot '{plot['filename']}' saved successfully in '{config_plots_dir}'.")
    
    # Example: Statistical Analysis (ANOVA) for Jain's GPU Fairness Index
    perform_anova(master_df, 'Jains_GPU_Fairness', config_plots_dir)
    
    # Similarly, perform ANOVA for Gini's GPU Index
    perform_anova(master_df, 'Ginis_GPU_Index', config_plots_dir)

def perform_anova(master_df, metric, output_dir):
    anova_df = master_df[['Utility_Function', metric]].dropna()
    groups = [group[metric].values for name, group in anova_df.groupby('Utility_Function')]
    
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        f_val, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA Results for {metric.replace('_', ' ')}:")
        print(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            result = "Significant differences found between utility functions."
        else:
            result = "No significant differences found between utility functions."
        print(result)
        
        # Save ANOVA results to a text file
        anova_results_path = os.path.join(output_dir, f'anova_{metric}.txt')
        with open(anova_results_path, 'w') as f:
            f.write(f"ANOVA Results for {metric.replace('_', ' ')}:\n")
            f.write(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}\n")
            f.write(result + "\n")
        print(f"ANOVA results saved to '{anova_results_path}'.")
    else:
        print(f"\nNot enough data for ANOVA on {metric.replace('_', ' ')}.")

def generate_summary(master_df, config):
    name = config['name']
    plot_subdir = config['plot_subdir']
    config_plots_dir = os.path.join(base_plots_directory, plot_subdir)
    
    if master_df.empty:
        print(f"No data available for summary in configuration '{name}'.")
        return
    
    print(f"\nAggregated Metrics for Configuration '{name}':")
    print(master_df)
    
    # Save the master DataFrame to a CSV file
    summary_csv_path = os.path.join(config_plots_dir, 'aggregated_metrics_summary.csv')
    master_df.to_csv(summary_csv_path, index=False)
    print(f"Aggregated metrics summary saved to '{summary_csv_path}'.")

# ------------------------------
# 4. Main Execution Loop
# ------------------------------

def main():
    for config in file_configurations:
        print(f"\nProcessing configuration: {config['name']}")
        
        # Process files and get aggregated metrics
        master_df = process_files_for_configuration(config)
        
        # Generate plots if there is data
        if not master_df.empty:
            plot_metrics(master_df, config, config['plot_subdir'])
        
        # Generate summary
        generate_summary(master_df, config)

if __name__ == "__main__":
    main()
