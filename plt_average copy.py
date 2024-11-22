import os
import re
import ast
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Configuration Variables
# -------------------------

# Plot size (width, height) in inches
PLOT_WIDTH = 12
PLOT_HEIGHT = 8

# Text size for labels and legends
TEXT_SIZE = 14

# Job ID range to consider
JOB_ID_START = 100
JOB_ID_END = 140

# Minimum average number of nodes required
MIN_AVG_NODES = 2

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the directory containing your CSV files
DATA_DIRECTORY = '.'  # Adjust the path if necessary

# Define the directory to save plots
PLOTS_DIRECTORY = 'plots'
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

# Define the filename pattern for main files
MAIN_FILE_REGEX = re.compile(
    r'^(\d+)_70J_50N_NFD_HN_NDJ_(SPS_BW|SPS_NBW|MPS_BW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO_jobs_report\.csv$'
)

# Define the utilities to process
SELECTED_UTILITIES = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']

# -------------------------
# Helper Function for Legend Customization
# -------------------------

def customize_legend(ax, text_size, title_text_size):
    """
    Customize the legend to be arranged over two lines.

    Parameters:
    - ax: Matplotlib Axes object
    - text_size: Font size for legend labels
    - title_text_size: Font size for legend title
    """
    lines, labels = ax.get_legend_handles_labels()
    num_labels = len(labels)
    # Calculate number of columns to have two rows
    ncol = math.ceil(num_labels / 2)
    ax.legend(
        lines, labels,
        fontsize=text_size,
        title_fontsize=title_text_size,
        loc='upper center',
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False  # Optional: Remove the legend frame for cleaner look
    )

# -------------------------
# Data Collection and Processing
# -------------------------

# List all main CSV files matching the pattern
file_list = [f for f in os.listdir(DATA_DIRECTORY) if MAIN_FILE_REGEX.match(f)]

if not file_list:
    print("No main CSV files found matching the pattern in the specified directory.")
    exit(1)

print(f"Found {len(file_list)} main files to process.")

# Initialize an empty list to collect DataFrames
data_frames = []

for filename in file_list:
    # Extract the job ID, allocation type, and utility from the filename
    match = MAIN_FILE_REGEX.match(filename)
    if match:
        job_id_str, allocation_type, utility = match.groups()
        job_id = int(job_id_str)
        
        # Filter based on job ID range
        if not (JOB_ID_START <= job_id <= JOB_ID_END):
            print(f"Skipping file {filename} with job ID {job_id} outside the range {JOB_ID_START}-{JOB_ID_END}.")
            continue

        if utility in SELECTED_UTILITIES:
            # Read the CSV file
            file_path = os.path.join(DATA_DIRECTORY, filename)
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                print(f"Error parsing {filename}. Skipping this file.")
                continue

            # Handle potential unnamed columns
            if df.columns[0] == '':
                df.columns = df.columns[1:].tolist() + ['Extra']
                df = df.iloc[:, :-1]

            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])

            # Ensure required columns are present
            required_columns = ['exec_time', 'submit_time', 'duration', 'num_pod', 'num_gpu', 'final_node_allocation']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns {missing_columns} in {filename}. Skipping this file.")
                continue

            # Convert columns to numeric where applicable
            numeric_columns = ['exec_time', 'submit_time', 'duration', 'num_pod', 'num_gpu']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing values in required numeric columns
            df = df.dropna(subset=numeric_columns)

            # Parse 'final_node_allocation' and compute 'num_nodes'
            def compute_num_nodes(allocation_str):
                try:
                    # Safely evaluate the string to a list
                    allocation_list = ast.literal_eval(allocation_str)
                    if isinstance(allocation_list, list):
                        return len(set(allocation_list))
                    else:
                        return None
                except (ValueError, SyntaxError):
                    return None

            df['num_nodes'] = df['final_node_allocation'].apply(compute_num_nodes)

            # Drop rows where 'num_nodes' couldn't be computed
            df = df.dropna(subset=['num_nodes'])

            # Ensure 'num_nodes' is integer
            df['num_nodes'] = df['num_nodes'].astype(int)

            # Calculate the metric: (exec_time - submit_time) / duration
            df['metric'] = (df['exec_time'] - df['submit_time']) / df['duration']

            # Add the 'utility' and 'allocation_type' columns
            df['utility'] = utility
            df['allocation_type'] = allocation_type

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

# Define the allocation types
allocation_types = ['SPS_BW', 'SPS_NBW', 'MPS_BW']

# Set global font sizes
plt.rcParams.update({
    'axes.titlesize': TEXT_SIZE,
    'axes.labelsize': TEXT_SIZE,
    'legend.fontsize': TEXT_SIZE,
    'legend.title_fontsize': TEXT_SIZE,
    'xtick.labelsize': TEXT_SIZE,
    'ytick.labelsize': TEXT_SIZE
})

# -------------------------
# Plotting
# -------------------------

for allocation in allocation_types:
    # Filter data for the current allocation type
    alloc_data = all_data[all_data['allocation_type'] == allocation]

    if alloc_data.empty:
        print(f"No data found for allocation type {allocation}. Skipping plots for this type.")
        continue

    # -------------------------
    # Plot 1: Average Metric vs. Number of Pods
    # -------------------------
    # Compute the average metric grouped by utility and num_pod
    avg_pod_metric = alloc_data.groupby(['utility', 'num_pod'])['metric'].mean().reset_index()

    # Line plot of average metric vs. num_pod, colored by utility
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    sns.lineplot(data=avg_pod_metric, x='num_pod', y='metric', hue='utility', marker='o')

    # Set plot labels
    plt.xlabel('Number of Pods')
    plt.ylabel('Average Normalized Job Completion Time')
    # Remove the title
    # plt.title(f'Average Job Completion Time vs Number of Pods by Utility ({allocation})')

    # Customize legend
    ax = plt.gca()
    customize_legend(ax, TEXT_SIZE, TEXT_SIZE)

    plt.tight_layout()

    # Save the plot
    avg_pod_plot_path = os.path.join(PLOTS_DIRECTORY, f'metric_vs_num_pod_average_{allocation}.png')
    plt.savefig(avg_pod_plot_path)
    plt.close()

    # -------------------------
    # Plot 2: Average Metric vs. Number of GPUs
    # -------------------------
    # Compute the average metric grouped by utility and num_gpu
    avg_gpu_metric = alloc_data.groupby(['utility', 'num_gpu'])['metric'].mean().reset_index()

    # Line plot of average metric vs. num_gpu, colored by utility
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    sns.lineplot(data=avg_gpu_metric, x='num_gpu', y='metric', hue='utility', marker='o')

    # Set plot labels
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Normalized Job Completion Time')
    # Remove the title
    # plt.title(f'Average Job Completion Time vs Number of GPUs by Utility ({allocation})')

    # Customize legend
    ax = plt.gca()
    customize_legend(ax, TEXT_SIZE, TEXT_SIZE)

    plt.tight_layout()

    # Save the plot
    avg_gpu_plot_path = os.path.join(PLOTS_DIRECTORY, f'metric_vs_num_gpu_average_{allocation}.png')
    plt.savefig(avg_gpu_plot_path)
    plt.close()

    # -------------------------
    # Plot 3: Average Number of Nodes vs. Number of Pods
    # -------------------------
    # Compute the average number of nodes grouped by utility and num_pod
    avg_pod_nodes = alloc_data.groupby(['utility', 'num_pod'])['num_nodes'].mean().reset_index()

    # Filter groups with average num_nodes >= MIN_AVG_NODES
    avg_pod_nodes_filtered = avg_pod_nodes[avg_pod_nodes['num_nodes'] >= MIN_AVG_NODES]

    if avg_pod_nodes_filtered.empty:
        print(f"No groups with average number of nodes >= {MIN_AVG_NODES} for allocation type {allocation}. Skipping Plot 3.")
    else:
        # Line plot of average number of nodes vs. num_pod, colored by utility
        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        sns.lineplot(data=avg_pod_nodes_filtered, x='num_pod', y='num_nodes', hue='utility', marker='o')

        # Set plot labels
        plt.xlabel('Number of Pods')
        plt.ylabel('Average Number of Nodes')
        # Remove the title
        # plt.title(f'Average Number of Nodes vs Number of Pods by Utility ({allocation})')

        # Customize legend
        ax = plt.gca()
        customize_legend(ax, TEXT_SIZE, TEXT_SIZE)

        plt.tight_layout()

        # Save the plot
        avg_pod_nodes_plot_path = os.path.join(PLOTS_DIRECTORY, f'nodes_vs_num_pod_average_{allocation}.png')
        plt.savefig(avg_pod_nodes_plot_path)
        plt.close()

    # -------------------------
    # Plot 4: Average Number of Nodes vs. Number of GPUs
    # -------------------------
    # Compute the average number of nodes grouped by utility and num_gpu
    avg_gpu_nodes = alloc_data.groupby(['utility', 'num_gpu'])['num_nodes'].mean().reset_index()

    # Filter groups with average num_nodes >= MIN_AVG_NODES
    avg_gpu_nodes_filtered = avg_gpu_nodes[avg_gpu_nodes['num_nodes'] >= MIN_AVG_NODES]

    if avg_gpu_nodes_filtered.empty:
        print(f"No groups with average number of nodes >= {MIN_AVG_NODES} for allocation type {allocation}. Skipping Plot 4.")
    else:
        # Line plot of average number of nodes vs. num_gpu, colored by utility
        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        sns.lineplot(data=avg_gpu_nodes_filtered, x='num_gpu', y='num_nodes', hue='utility', marker='o')

        # Set plot labels
        plt.xlabel('Number of GPUs')
        plt.ylabel('Average Number of Nodes')
        # Remove the title
        # plt.title(f'Average Number of Nodes vs Number of GPUs by Utility ({allocation})')

        # Customize legend
        ax = plt.gca()
        customize_legend(ax, TEXT_SIZE, TEXT_SIZE)

        plt.tight_layout()

        # Save the plot
        avg_gpu_nodes_plot_path = os.path.join(PLOTS_DIRECTORY, f'nodes_vs_num_gpu_average_{allocation}.png')
        plt.savefig(avg_gpu_nodes_plot_path)
        plt.close()

print("All plots have been saved in the 'plots' directory.")
