import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the directory containing your CSV files
data_directory = '.'  # Adjust the path if necessary

# Define the directory to save plots
plots_directory = 'plots'
os.makedirs(plots_directory, exist_ok=True)

# Define the filename pattern for main files and allocations files
main_file_regex = re.compile(
    r'(\d+)_70J_50N_NFD_HN_NDJ_(?:SPS_BW|SPS_NBW|MPS_BW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO_jobs_report\.csv'
)

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
selected_reps = range(1, 40)  # Replications 1 to 39

# Define the number of intervals
num_intervals = 2  # Change this value to 1, 2, 4, etc., as needed

# List all main CSV files matching the pattern
file_list = [f for f in os.listdir(data_directory) if main_file_regex.match(f)]

if not file_list:
    print("No main CSV files found matching the pattern in the specified directory.")
    exit(1)

print(f"Found {len(file_list)} main files to process.")

# Initialize an empty list to collect DataFrames
data_frames = []

for filename in file_list:
    # Extract the utility from the filename
    match = main_file_regex.match(filename)
    if match:
        utility = match.group(2)
        if utility in selected_utilities:
            # Read the CSV file
            file_path = os.path.join(data_directory, filename)
            try:
                # Attempt to read the CSV file with the correct header
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                print(f"Error parsing {filename}. Skipping this file.")
                continue

            # Check for an extra unnamed column at the beginning
            if df.columns[0] == '':
                # Shift the columns to the left
                df.columns = df.columns[1:].tolist() + ['Extra']
                df = df.iloc[:, :-1]

            # Alternatively, drop the unnamed column if it's named 'Unnamed: 0'
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])

            # Ensure that the required columns are present
            required_columns = ['exec_time', 'submit_time', 'duration', 'num_pod', 'num_gpu']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns {missing_columns} in {filename}. Skipping this file.")
                continue

            # Convert columns to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing values in required columns
            df = df.dropna(subset=required_columns)

            # Calculate the metric: (exec_time - submit_time) / duration
            df['metric'] = (df['exec_time'] - df['submit_time']) / df['duration']

            # Add the 'utility' column
            df['utility'] = utility

            # Append the DataFrame to the list
            data_frames.append(df)

if data_frames:
    # Concatenate all DataFrames
    all_data = pd.concat(data_frames, ignore_index=True)
else:
    print("No data to process after reading files.")
    exit(1)

# Ensure 'num_pod' and 'num_gpu' are integers
all_data['num_pod'] = all_data['num_pod'].astype(int)
all_data['num_gpu'] = all_data['num_gpu'].astype(int)

# Scatter plot of metric vs. num_pod, colored by utility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_data, x='num_pod', y='metric', hue='utility', alpha=0.7)

# Set plot labels and title
plt.xlabel('Number of Pods')
plt.ylabel('Normalized Job Completion Time')
plt.title('Job Completion Time vs Number of Pods by Utility')
plt.legend(title='Utility')
plt.tight_layout()

# Save the plot
scatter_plot_path = os.path.join(plots_directory, 'metric_vs_num_pod_scatter.png')
plt.savefig(scatter_plot_path)
plt.close()

# *** Replace Boxplot with Average Plot for num_pod ***

# Compute the average metric grouped by utility and num_pod
avg_pod = all_data.groupby(['utility', 'num_pod'])['metric'].mean().reset_index()

# Line plot of average metric vs. num_pod, colored by utility
plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_pod, x='num_pod', y='metric', hue='utility', marker='o')

# Set plot labels and title
plt.xlabel('Number of Pods')
plt.ylabel('Average Normalized Job Completion Time')
plt.title('Average Job Completion Time vs Number of Pods by Utility')
plt.legend(title='Utility')
plt.tight_layout()

# Save the plot
avg_pod_plot_path = os.path.join(plots_directory, 'metric_vs_num_pod_average.png')
plt.savefig(avg_pod_plot_path)
plt.close()

# Scatter plot of metric vs. num_gpu, colored by utility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_data, x='num_gpu', y='metric', hue='utility', alpha=0.7)

# Set plot labels and title
plt.xlabel('Number of GPUs')
plt.ylabel('Normalized Job Completion Time')
plt.title('Job Completion Time vs Number of GPUs by Utility')
plt.legend(title='Utility')
plt.tight_layout()

# Save the plot
scatter_plot_gpu_path = os.path.join(plots_directory, 'metric_vs_num_gpu_scatter.png')
plt.savefig(scatter_plot_gpu_path)
plt.close()

# *** Replace Boxplot with Average Plot for num_gpu ***

# Compute the average metric grouped by utility and num_gpu
avg_gpu = all_data.groupby(['utility', 'num_gpu'])['metric'].mean().reset_index()

# Line plot of average metric vs. num_gpu, colored by utility
plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_gpu, x='num_gpu', y='metric', hue='utility', marker='o')

# Set plot labels and title
plt.xlabel('Number of GPUs')
plt.ylabel('Average Normalized Job Completion Time')
plt.title('Average Job Completion Time vs Number of GPUs by Utility')
plt.legend(title='Utility')
plt.tight_layout()

# Save the plot
avg_gpu_plot_path = os.path.join(plots_directory, 'metric_vs_num_gpu_average.png')
plt.savefig(avg_gpu_plot_path)
plt.close()

# *** Add Heatmap to Characterize Jobs Distribution ***
# Create a pivot table with counts of jobs
heatmap_data = all_data.groupby(['num_pod', 'num_gpu']).size().reset_index(name='job_count')

# Pivot the data to have num_pod as rows and num_gpu as columns
heatmap_pivot = heatmap_data.pivot(index='num_pod', columns='num_gpu', values='job_count')

# Replace NaN with 0 for combinations with no jobs
heatmap_pivot = heatmap_pivot.fillna(0)

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_pivot, cmap='YlGnBu', annot=True, fmt=".0f", linewidths=.5)

# Set plot labels and title
plt.xlabel('Number of GPUs')
plt.ylabel('Number of Pods')
plt.title('Heatmap of Job Distribution by Number of Pods and GPUs')

plt.tight_layout()

# Save the heatmap
heatmap_path = os.path.join(plots_directory, 'jobs_distribution_heatmap.png')
plt.savefig(heatmap_path)
plt.close()

# *** Add Joint Plot to Characterize Jobs Distribution ***
# Create a joint plot
joint_plot = sns.jointplot(
    data=all_data,
    x='num_pod',
    y='num_gpu',
    kind='hex',  # Options: 'scatter', 'kde', 'hex'
    color='teal',
    gridsize=30,
    marginal_kws=dict(bins=30, fill=True)
)

# Set plot labels and title
joint_plot.set_axis_labels('Number of Pods', 'Number of GPUs')
plt.suptitle('Joint Plot of Job Distribution by Number of Pods and GPUs', y=1.02)

# Save the joint plot
joint_plot_path = os.path.join(plots_directory, 'jobs_distribution_jointplot.png')
plt.savefig(joint_plot_path)
plt.close()

print("Average and distribution characterization plots have been saved in the 'plots' directory.")
