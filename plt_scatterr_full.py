import os
import re
import ast  # To safely parse string representations of lists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Configuration --------------------

# Optional: Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the directory containing your CSV files
data_directory = '.'  # Adjust the path if necessary

# Define the main directory to save plots
plots_directory = 'plots'
os.makedirs(plots_directory, exist_ok=True)

# Define the filename pattern with capturing groups for experiment type and utility
# Group 1: Replication ID
# Group 2: Experiment Type (SPS_BW, SPS_NBW, MPS_BW)
# Group 3: Utility (TETRIS, DRF, etc.)
main_file_regex = re.compile(
    r'^(\d+)_70J_50N_NFD_HN_NDJ_(SPS_BW|SPS_NBW|MPS_BW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO_jobs_report\.csv$'
)

# Define the utilities and replications to process
selected_utilities = ['TETRIS', 'DRF', 'LIKELIHOOD', 'SGF', 'LGF', 'SEQ']
selected_reps = range(1, 40)  # Replications 1 to 39

# Define GPU bins for categorization (e.g., 0-50, 51-100, etc.)
gpu_bin_edges = [0, 50, 100, 200, 300, np.inf]
gpu_bin_labels = ['50', '100', '200', '300', '800']

# -------------------- Data Processing --------------------

# List all main CSV files matching the pattern
file_list = [f for f in os.listdir(data_directory) if main_file_regex.match(f)]

if not file_list:
    print("No main CSV files found matching the pattern in the specified directory.")
    exit(1)

print(f"Found {len(file_list)} main files to process.")

# Initialize an empty list to collect DataFrames
data_frames = []

for filename in file_list:
    # Extract the replication ID, experiment type, and utility from the filename
    match = main_file_regex.match(filename)
    if match:
        replication_id = match.group(1)       # e.g., '1', '2', etc.
        experiment_type = match.group(2)      # SPS_BW, SPS_NBW, MPS_BW
        utility = match.group(3)              # TETRIS, DRF, etc.

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
            required_columns = ['complete_time', 'submit_time', 'duration', 'num_pod', 'num_gpu', 'final_node_allocation']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns {missing_columns} in {filename}. Skipping this file.")
                continue

            # Convert columns to numeric where applicable
            for col in ['complete_time', 'submit_time', 'duration', 'num_pod', 'num_gpu']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing values in required columns
            df = df.dropna(subset=['complete_time', 'submit_time', 'duration', 'num_pod', 'num_gpu'])

            # Calculate the metric: (complete_time - submit_time) / duration
            df['metric'] = (df['complete_time'] - df['submit_time']) / df['duration']

            # Parse the 'final_node_allocation' column and compute 'num_unique_nodes'
            def compute_unique_nodes(allocation_str):
                try:
                    # Safely parse the string to a list
                    allocation_list = ast.literal_eval(allocation_str)
                    # Ensure it's a list
                    if isinstance(allocation_list, list):
                        return len(set(allocation_list))
                    else:
                        return np.nan
                except (ValueError, SyntaxError):
                    return np.nan

            df['num_unique_nodes'] = df['final_node_allocation'].apply(compute_unique_nodes)
            # Define the filtering conditions
            # filtered_df = df[(df['num_unique_nodes'] > 2) & (df['num_gpu'] == 50)]

            # # Select specific columns to print
            # print('50')
            # print(filtered_df[['num_pod', 'final_node_allocation']])

            filtered_df = df[(df['num_unique_nodes'] > 2) & (df['num_gpu'] == 800)]
            if len(filtered_df):
                # Iterate over the rows of the DataFrame
                for index, row in filtered_df.iterrows():
                    print(f"num_pod: {row['num_pod']}, num_unique_nodes: {row['num_unique_nodes']}, final_node_allocation: {row['final_node_allocation']}")


            # Drop rows with missing 'num_unique_nodes'
            df = df.dropna(subset=['num_unique_nodes'])

            # Convert 'num_unique_nodes' to integer
            df['num_unique_nodes'] = df['num_unique_nodes'].astype(int)

            # Filter to include only jobs with at least two unique node allocations
            df = df[df['num_unique_nodes'] >= 2]

            # Add the 'experiment_type', 'utility', and 'replication_id' columns
            df['experiment_type'] = experiment_type
            df['utility'] = utility
            df['replication_id'] = replication_id  # Optional: if you need to track replication

            # Categorize 'num_gpu' into bins
            df['gpu_bin'] = pd.cut(df['num_gpu'], bins=gpu_bin_edges, labels=gpu_bin_labels, right=False)

            # Drop rows with missing 'gpu_bin' (if any)
            df = df.dropna(subset=['gpu_bin'])

            # Append the DataFrame to the list
            data_frames.append(df)

if data_frames:
    # Concatenate all DataFrames
    all_data = pd.concat(data_frames, ignore_index=True)
else:
    print("No data to process after reading files.")
    exit(1)

# Ensure 'num_pod', 'num_gpu', and 'num_unique_nodes' are integers
all_data['num_pod'] = all_data['num_pod'].astype(int)
all_data['num_gpu'] = all_data['num_gpu'].astype(int)
all_data['num_unique_nodes'] = all_data['num_unique_nodes'].astype(int)

print(f"Total jobs after filtering: {len(all_data)}")

# -------------------- Plotting Section --------------------

# Define the list of experiment types
experiment_types = ['SPS_BW', 'SPS_NBW', 'MPS_BW']

for exp_type in experiment_types:
    # Filter data for the current experiment type
    exp_data = all_data[all_data['experiment_type'] == exp_type]

    if exp_data.empty:
        print(f"No data available for experiment type '{exp_type}'. Skipping plots for this type.")
        continue

    print(f"Generating plots for experiment type: {exp_type}")

    # Create a subdirectory for the current experiment type
    exp_plots_directory = os.path.join(plots_directory, exp_type)
    os.makedirs(exp_plots_directory, exist_ok=True)

    # -------------------- Metric vs. Number of Pods --------------------

    # 1. Scatter plot of metric vs. num_pod, colored by utility
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=exp_data, x='num_pod', y='metric', hue='utility', alpha=0.7)

    # Set plot labels and title
    plt.xlabel('Number of Pods')
    plt.ylabel('Normalized Job Completion Time')
    plt.title(f'Job Completion Time vs Number of Pods by Utility ({exp_type})')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    scatter_plot_path = os.path.join(exp_plots_directory, f'metric_vs_num_pod_scatter_{exp_type}.png')
    plt.savefig(scatter_plot_path)
    plt.close()

    # 2. Boxplot of metric vs. num_pod, grouped by utility without outliers
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=exp_data, x='num_pod', y='metric', hue='utility', showfliers=False)

    # Set plot labels and title
    plt.xlabel('Number of Pods')
    plt.ylabel('Normalized Job Completion Time')
    plt.title(f'Job Completion Time vs Number of Pods by Utility ({exp_type})\n(Outliers Removed)')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    boxplot_pod_path = os.path.join(exp_plots_directory, f'metric_vs_num_pod_boxplot_no_outliers_{exp_type}.png')
    plt.savefig(boxplot_pod_path)
    plt.close()

    # -------------------- Metric vs. Number of GPUs --------------------

    # 3. Scatter plot of metric vs. num_gpu, colored by utility
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=exp_data, x='num_gpu', y='metric', hue='utility', alpha=0.7)

    # Set plot labels and title
    plt.xlabel('Number of GPUs')
    plt.ylabel('Normalized Job Completion Time')
    plt.title(f'Job Completion Time vs Number of GPUs by Utility ({exp_type})')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    scatter_plot_gpu_path = os.path.join(exp_plots_directory, f'metric_vs_num_gpu_scatter_{exp_type}.png')
    plt.savefig(scatter_plot_gpu_path)
    plt.close()

    # 4. Boxplot of metric vs. num_gpu, grouped by utility without outliers (GPUs >= 50)
    # Filter data to include only num_gpu >= 50
    gpu_filtered_data = exp_data[exp_data['num_gpu'] >= 50]

    if gpu_filtered_data.empty:
        print(f"No data available with 'num_gpu' >= 50 for experiment type '{exp_type}'. Skipping GPU boxplot.")
    else:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=gpu_filtered_data, x='num_gpu', y='metric', hue='utility', showfliers=False)

        # Set plot labels and title
        plt.xlabel('Number of GPUs')
        plt.ylabel('Normalized Job Completion Time')
        plt.title(f'Job Completion Time vs Number of GPUs by Utility ({exp_type})\n(GPUs >= 50, Outliers Removed)')
        plt.legend(title='Utility')
        plt.tight_layout()

        # Save the plot
        boxplot_gpu_path = os.path.join(exp_plots_directory, f'metric_vs_num_gpu_boxplot_no_outliers_ge_50_{exp_type}.png')
        plt.savefig(boxplot_gpu_path)
        plt.close()

    # -------------------- Histograms with KDE --------------------

    # 5. Histograms for Job Distribution: Number of Pods with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=exp_data,
        x='num_pod',
        bins=range(exp_data['num_pod'].min(), exp_data['num_pod'].max() + 2),
        kde=True,  # Add KDE to resemble Gaussian distribution
        stat='density',  # Normalize the histogram
        color='skyblue'
    )

    # Set plot labels and title
    plt.xlabel('Number of Pods')
    plt.ylabel('Density')
    plt.title(f'Distribution of Jobs by Number of Pods with KDE ({exp_type})')
    plt.tight_layout()

    # Save the histogram
    hist_pod_path = os.path.join(exp_plots_directory, f'jobs_distribution_num_pod_histogram_kde_{exp_type}.png')
    plt.savefig(hist_pod_path)
    plt.close()

    # 6. Histograms for Job Distribution: Number of GPUs with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=exp_data,
        x='num_gpu',
        bins=range(exp_data['num_gpu'].min(), exp_data['num_gpu'].max() + 2),
        kde=True,  # Add KDE to resemble Gaussian distribution
        stat='density',  # Normalize the histogram
        color='salmon'
    )

    # Set plot labels and title
    plt.xlabel('Number of GPUs')
    plt.ylabel('Density')
    plt.title(f'Distribution of Jobs by Number of GPUs with KDE ({exp_type})')
    plt.tight_layout()

    # Save the histogram
    hist_gpu_path = os.path.join(exp_plots_directory, f'jobs_distribution_num_gpu_histogram_kde_{exp_type}.png')
    plt.savefig(hist_gpu_path)
    plt.close()

    # --------------- New Plot Category: Relative Number of Required GPUs per Number of Pods ---------------

    # 7. Stacked Bar Plot: Relative Number of Required GPUs per Number of Pods
    # Calculate counts of GPU bins for each number of pods
    gpu_pod_counts = exp_data.groupby(['num_pod', 'gpu_bin']).size().reset_index(name='counts')

    # Pivot the data to have GPU bins as columns
    gpu_pod_pivot = gpu_pod_counts.pivot(index='num_pod', columns='gpu_bin', values='counts').fillna(0)

    # Calculate relative frequencies
    gpu_pod_rel = gpu_pod_pivot.div(gpu_pod_pivot.sum(axis=1), axis=0)

    # Sort the GPU bins for consistent color ordering
    gpu_pod_rel = gpu_pod_rel[gpu_bin_labels]

    # Plot stacked bar plot
    gpu_pod_rel.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')

    plt.xlabel('Number of Pods')
    plt.ylabel('Relative Frequency')
    plt.title(f'Relative Number of Required GPUs per Number of Pods ({exp_type})')
    plt.legend(title='Number of GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    gpu_pod_rel_plot_path = os.path.join(exp_plots_directory, f'relative_num_gpu_per_num_pod_{exp_type}.png')
    plt.savefig(gpu_pod_rel_plot_path)
    plt.close()

    # -------------------- New Metric: Number of Unique Nodes --------------------

    # 8. Scatter plot of num_unique_nodes vs. num_pod, colored by utility
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=exp_data, x='num_pod', y='num_unique_nodes', hue='utility', alpha=0.7)

    # Set plot labels and title
    plt.xlabel('Number of Pods')
    plt.ylabel('Number of Unique Nodes')
    plt.title(f'Number of Unique Nodes vs Number of Pods by Utility ({exp_type})')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    scatter_unique_pod_path = os.path.join(exp_plots_directory, f'unique_nodes_vs_num_pod_scatter_{exp_type}.png')
    plt.savefig(scatter_unique_pod_path)
    plt.close()

    # 9. Boxplot of num_unique_nodes vs. num_pod, grouped by utility without outliers
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=exp_data, x='num_pod', y='num_unique_nodes', hue='utility', showfliers=False)

    # Set plot labels and title
    plt.xlabel('Number of Pods')
    plt.ylabel('Number of Unique Nodes')
    plt.title(f'Number of Unique Nodes vs Number of Pods by Utility ({exp_type})\n(Outliers Removed)')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    boxplot_unique_pod_path = os.path.join(exp_plots_directory, f'unique_nodes_vs_num_pod_boxplot_no_outliers_{exp_type}.png')
    plt.savefig(boxplot_unique_pod_path)
    plt.close()

    # 10. Scatter plot of num_unique_nodes vs. num_gpu, colored by utility
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=exp_data, x='num_gpu', y='num_unique_nodes', hue='utility', alpha=0.7)

    # Set plot labels and title
    plt.xlabel('Number of GPUs')
    plt.ylabel('Number of Unique Nodes')
    plt.title(f'Number of Unique Nodes vs Number of GPUs by Utility ({exp_type})')
    plt.legend(title='Utility')
    plt.tight_layout()

    # Save the plot
    scatter_unique_gpu_path = os.path.join(exp_plots_directory, f'unique_nodes_vs_num_gpu_scatter_{exp_type}.png')
    plt.savefig(scatter_unique_gpu_path)
    plt.close()

    # 11. Boxplot of num_unique_nodes vs. num_gpu, grouped by utility without outliers (GPUs >= 50)
    if gpu_filtered_data.empty:
        print(f"No data available with 'num_gpu' >= 50 for experiment type '{exp_type}'. Skipping unique nodes GPU boxplot.")
    else:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=gpu_filtered_data, x='num_gpu', y='num_unique_nodes', hue='utility', showfliers=False)

        # Set plot labels and title
        plt.xlabel('Number of GPUs')
        plt.ylabel('Number of Unique Nodes')
        plt.title(f'Number of Unique Nodes vs Number of GPUs by Utility ({exp_type})\n(GPUs >= 50, Outliers Removed)')
        plt.legend(title='Utility')
        plt.tight_layout()

        # Save the plot
        boxplot_unique_gpu_path = os.path.join(exp_plots_directory, f'unique_nodes_vs_num_gpu_boxplot_no_outliers_ge_50_{exp_type}.png')
        plt.savefig(boxplot_unique_gpu_path)
        plt.close()

    # 12. Histograms for Job Distribution: Number of Unique Nodes with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=exp_data,
        x='num_unique_nodes',
        bins=range(exp_data['num_unique_nodes'].min(), exp_data['num_unique_nodes'].max() + 2),
        kde=True,  # Add KDE to resemble Gaussian distribution
        stat='density',  # Normalize the histogram
        color='purple'
    )

    # Set plot labels and title
    plt.xlabel('Number of Unique Nodes')
    plt.ylabel('Density')
    plt.title(f'Distribution of Jobs by Number of Unique Nodes with KDE ({exp_type})')
    plt.tight_layout()

    # Save the histogram
    hist_unique_nodes_path = os.path.join(exp_plots_directory, f'jobs_distribution_num_unique_nodes_histogram_kde_{exp_type}.png')
    plt.savefig(hist_unique_nodes_path)
    plt.close()

print("All plots (scatter, boxplots without outliers, histograms with KDE, and relative GPU allocations) have been saved in the 'plots' directory, organized by experiment type.")
