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

# Define the directory to save plots and aggregated data
plots_directory = 'plots'
os.makedirs(plots_directory, exist_ok=True)

# Define the filename pattern for main files and allocations files
# Main files: 150J_100N_NFD_HN_NDJ_NBW_rep_utility_FIFO.csv
# Allocations files: 150J_100N_NFD_HN_NDJ_NBW_rep_utility_FIFO_jobs_report.csv
main_file_regex = re.compile(
    r'50J_50N_NFD_HN_NDJ_BW_(\d+)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO\.csv')
alloc_file_suffix = '_jobs_report.csv'

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
selected_reps = range(1, 50)  # Replications 1 to 49

# List all main CSV files matching the pattern
file_list = [f for f in os.listdir(data_directory)
             if main_file_regex.match(f)]

if not file_list:
    print("No main CSV files found matching the pattern in the specified directory.")
    exit(1)

print(f"Found {len(file_list)} main files to process.")

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

# ------------------------------
# 3. Initialize Data Structures
# ------------------------------

# Initialize a list to store all job metrics
all_jobs_metrics = []

# ------------------------------
# 4. Process Each File
# ------------------------------

for filename in file_list:
    match = main_file_regex.match(filename)
    if not match:
        print(
            f"Filename '{filename}' does not match the expected pattern. Skipping.")
        continue
    rep = int(match.group(1))
    utility_function = match.group(2)

    # Filter based on selected utilities and replications
    if utility_function not in selected_utilities or rep not in selected_reps:
        print(
            f"Skipping file '{filename}' as it does not match selected utilities or replications.")
        continue

    filepath = os.path.join(data_directory, filename)

    # Determine the corresponding allocations file
    alloc_filename = filename.replace('.csv', alloc_file_suffix)
    alloc_filepath = os.path.join(data_directory, alloc_filename)

    if not os.path.exists(alloc_filepath):
        print(
            f"Allocations file '{alloc_filename}' not found for '{filename}'. Skipping.")
        continue

    # Load the allocations CSV file
    try:
        df_alloc = pd.read_csv(alloc_filepath)
        print(f"Processing allocations file: {alloc_filename}")
        # Uncomment the next line to debug columns
        # print("Columns in df_alloc:", df_alloc.columns.tolist())
    except Exception as e:
        print(f"Error loading allocations file '{alloc_filename}': {e}")
        continue

    # Adjust column names if necessary
    required_columns = ['job_id', 'submit_time',
                        'complete_time', 'num_gpu']
    df_alloc_standard = standardize_columns(df_alloc, required_columns)

    if df_alloc_standard is None:
        print(
            f"Skipping allocations file '{alloc_filename}' due to missing required columns.")
        continue

    # Proceed with standardized column names
    jobs_df = df_alloc_standard[['job_id', 'submit_time',
                                 'complete_time', 'num_gpu']].drop_duplicates().copy()

    # Ensure numeric columns are of numeric type
    jobs_df['submit_time'] = pd.to_numeric(
        jobs_df['submit_time'], errors='coerce')
    jobs_df['complete_time'] = pd.to_numeric(
        jobs_df['complete_time'], errors='coerce')
    jobs_df['num_gpu'] = pd.to_numeric(jobs_df['num_gpu'], errors='coerce')

    # Remove jobs with missing data
    jobs_df = jobs_df.dropna(subset=[
                             'job_id', 'submit_time', 'complete_time', 'num_gpu'])

    # Compute Job Completion Time (JCT)
    jobs_df['JCT'] = jobs_df['complete_time'] - jobs_df['submit_time']

    # Filter out jobs with non-positive JCT
    jobs_df = jobs_df[jobs_df['JCT'] >= 0]

    # Append utility function and replication info
    jobs_df['Utility_Function'] = utility_function
    jobs_df['Replication'] = rep

    # Collect all job data for later analysis
    all_jobs_metrics.append(jobs_df)

# ------------------------------
# 5. Combine All Jobs Data
# ------------------------------

if not all_jobs_metrics:
    print("No job data collected. Exiting.")
    exit(1)

# Concatenate all job data into a single DataFrame
all_jobs_df = pd.concat(all_jobs_metrics, ignore_index=True)

print(f"Total number of jobs collected: {len(all_jobs_df)}")

# ------------------------------
# 6. Categorize Jobs into Quartiles Based on GPU Requirement
# ------------------------------

# Assign quartiles using pd.qcut, without specifying labels
# Using labels=False to get integer labels
try:
    all_jobs_df['Quartile'], bins = pd.qcut(
        all_jobs_df['num_gpu'],
        q=4,
        labels=False,
        retbins=True,
        duplicates='drop'
    )
except ValueError as e:
    print(f"Error in quartile assignment: {e}")
    exit(1)

# Extract unique bin edges
unique_bins = np.unique(bins)

# Calculate the number of quartiles assigned
num_quartiles = len(unique_bins) - 1  # number of intervals

print(f"Number of quartiles assigned: {num_quartiles}")

# Define quartile labels based on bins
quartile_labels = []
labels_available = []

for i in range(num_quartiles):
    quartile_labels.append(
        f"Q{i+1} ({unique_bins[i]:.1f} - {unique_bins[i+1]:.1f} GPUs)"
    )
    labels_available.append(f"Q{i+1}")

# Create mapping from labels_available to quartile_labels
quartile_mapping = dict(zip(labels_available, quartile_labels))

# Map quartiles to labels
all_jobs_df['Quartile'] = all_jobs_df['Quartile'].astype(int).map(
    lambda x: labels_available[x] if x < num_quartiles else np.nan)

all_jobs_df['Quartile_Range'] = all_jobs_df['Quartile'].map(quartile_mapping)

# ------------------------------
# 7. Analyze JCT Differences Between Quartiles Across Utility Functions
# ------------------------------

# Group by Utility Function and Quartile, then compute aggregate statistics
jct_stats = all_jobs_df.groupby(
    ['Utility_Function', 'Quartile']).agg(
    JCT_Mean=('JCT', 'mean'),
    JCT_Median=('JCT', 'median'),
    JCT_STD=('JCT', 'std'),
    Job_Count=('JCT', 'count')
).reset_index()

# Merge quartile range labels
jct_stats['Quartile_Range'] = jct_stats['Quartile'].map(quartile_mapping)

# Handle cases where Quartile_Range might be missing
jct_stats['Quartile_Range'] = jct_stats['Quartile_Range'].fillna(
    jct_stats['Quartile'])

# ------------------------------
# 8. Visualization
# ------------------------------

# Set the plotting style
sns.set(style="whitegrid")

# 8.a. CDF Plot: Job Completion Time (JCT) by Quartile and Utility Function
def plot_jct_cdf(df, title, filename):
    plt.figure(figsize=(14, 8))
    utility_functions = df['Utility_Function'].unique()
    quartiles = sorted(df['Quartile_Range'].unique())

    for utility in utility_functions:
        plt.figure(figsize=(14, 8))
        subset = df[df['Utility_Function'] == utility]
        for quartile in quartiles:
            quartile_subset = subset[subset['Quartile_Range'] == quartile]
            if quartile_subset.empty:
                continue
            sorted_jct = np.sort(quartile_subset['JCT'])
            cdf = np.arange(1, len(sorted_jct)+1) / len(sorted_jct)
            plt.plot(sorted_jct, cdf, label=quartile)
        plt.title(f"{title} for Utility Function: {utility}", fontsize=16)
        plt.xlabel('Job Completion Time (JCT)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        plt.legend(title='GPU Requirement Quartiles', fontsize=12, title_fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename.replace("{utility}", utility))
        plt.close()
        print(f"CDF plot for utility '{utility}' saved as '{filename.replace('{utility}', utility)}'.")

# Prepare the plotting function
def plot_jct_cdf_master(df, title_template, filename_template):
    utility_functions = df['Utility_Function'].unique()
    for utility in utility_functions:
        subset = df[df['Utility_Function'] == utility]
        quartiles = subset['Quartile_Range'].unique()
        plt.figure(figsize=(14, 8))
        for quartile in quartiles:
            quartile_subset = subset[subset['Quartile_Range'] == quartile]
            if quartile_subset.empty:
                continue
            sorted_jct = np.sort(quartile_subset['JCT'])
            cdf = np.arange(1, len(sorted_jct)+1) / len(sorted_jct)
            plt.plot(sorted_jct, cdf, label=quartile)
        plt.title(title_template.format(utility=utility), fontsize=16)
        plt.xlabel('Job Completion Time (JCT)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        plt.legend(title='GPU Requirement Quartiles', fontsize=12, title_fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plot_filename = filename_template.format(utility=utility)
        plt.savefig(plot_filename)
        plt.close()
        print(f"CDF plot for utility '{utility}' saved as '{plot_filename}'.")

# 8.a. Plot the CDFs
plot_jct_cdf_master(
    df=all_jobs_df,
    title_template='CDF of Job Completion Time (JCT) by Quartile for Utility Function: {utility}',
    filename_template=os.path.join(plots_directory, 'jct_cdf_by_quartile_{utility}.png')
)

# 8.b. Histogram: Distribution of num_gpu Across All Jobs
def plot_num_gpu_distribution(df, title, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_gpu'], bins=50, kde=True, color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel('Number of GPUs Requested', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved as '{filename}'.")

# Plot the distribution of num_gpu
plot_num_gpu_distribution(
    df=all_jobs_df,
    title='Distribution of GPU Requirements Across All Jobs',
    filename=os.path.join(
        plots_directory, 'num_gpu_distribution.png')
)

# ------------------------------
# 9. Save the Aggregated Metrics
# ------------------------------

# Save the JCT statistics to a CSV file
jct_stats_csv_path = os.path.join(
    plots_directory, 'jct_stats_by_quartile.csv')
jct_stats.to_csv(jct_stats_csv_path, index=False)
print(f"JCT statistics saved to '{jct_stats_csv_path}'.")

# Save all job data with quartile assignments to a CSV file
all_jobs_csv_path = os.path.join(
    plots_directory, 'all_jobs_with_quartiles.csv')
all_jobs_df.to_csv(all_jobs_csv_path, index=False)
print(f"All job data with quartiles saved to '{all_jobs_csv_path}'.")

# ------------------------------
# 10. Summary of Results
# ------------------------------

# Display the JCT statistics
print("\nJob Completion Time (JCT) Statistics by Quartile and Utility Function:")
print(jct_stats)

# Optionally, display the first few rows of all_jobs_df
# print("\nFirst few rows of all_jobs_df:")
# print(all_jobs_df.head())
