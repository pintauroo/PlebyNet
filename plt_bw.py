import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
import sys
import warnings

# Suppress pandas PerformanceWarnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Define a regex pattern to capture experiment set and policy
file_pattern = re.compile(
    r'(\d+)_70J_50N_NFD_HN_NDJ_(SPS|MPS)_(BW|NBW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO_topo\.csv'
)

def categorize_files(directory):
    """
    Categorize CSV files based on predefined regex pattern.

    Args:
        directory (str): Path to the directory containing CSV files.

    Returns:
        dict: Nested dictionary with experiment sets, policies, and lists of file paths.
    """
    categorized = {}
    all_files = glob.glob(os.path.join(directory, '*.csv'))

    if not all_files:
        print(f"No CSV files found in directory: {directory}")
        sys.exit(1)

    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = file_pattern.match(filename)
        if match:
            experiment_set = f"{match.group(2)}_{match.group(3)}"  # e.g., 'SPS_BW', 'MPS_BW', 'SPS_NBW'
            policy = match.group(4)  # 'TETRIS', 'DRF', etc.
            categorized.setdefault(experiment_set, {}).setdefault(policy, []).append(filepath)
            print(f"Categorized file '{filename}' under experiment '{experiment_set}', policy '{policy}'.")
        else:
            print(f"Warning: File '{filename}' did not match any known patterns and will be skipped.")

    if not categorized:
        print("Error: No files were categorized. Please check your regex patterns and file naming conventions.")
        sys.exit(1)

    return categorized

def compute_metrics(data):
    """
    Compute various network metrics from the given DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing network bandwidth data.

    Returns:
        dict: A dictionary containing computed metrics.
    """
    # Check if 'allocation_step' exists
    if 'allocation_step' not in data.columns:
        raise ValueError("Column 'allocation_step' not found in data.")

    # Set 'allocation_step' as the index
    data.set_index('allocation_step', inplace=True)

    # Identify reserved bandwidth columns
    reserved_bw_columns = [col for col in data.columns if 'reserved_bw' in col]
    if not reserved_bw_columns:
        raise ValueError("No 'reserved_bw' columns found in data.")

    # Identify total bandwidth columns corresponding to reserved_bw
    total_bw_columns = [col.replace('reserved_bw', 'total_bw') for col in reserved_bw_columns]

    # Check if total_bw_columns exist
    for tb_col in total_bw_columns:
        if tb_col not in data.columns:
            raise ValueError(f"Expected total bandwidth column '{tb_col}' not found in data.")

    # Identify available bandwidth columns
    available_bw_columns = [col for col in data.columns if 'available_bw' in col]
    if not available_bw_columns:
        raise ValueError("No 'available_bw' columns found in data.")

    # Calculate required metrics
    # Prepare a dictionary to hold new columns
    new_columns = {}

    # Total Reserved Bandwidth
    new_columns['total_reserved_bw'] = data[reserved_bw_columns].sum(axis=1)

    # Total Available Bandwidth
    new_columns['total_available_bw'] = data[available_bw_columns].sum(axis=1)

    # Calculate utilization (reserved / total)
    reserved_bw_data = data[reserved_bw_columns]
    total_bw_data = data[total_bw_columns]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        utilization = reserved_bw_data.div(total_bw_data)
        utilization = utilization.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    new_columns['average_utilization'] = utilization.mean(axis=1)

    # Fairness Index (JFI) calculation
    sum_reserved_bw = new_columns['total_reserved_bw']
    sum_reserved_bw_sq = (reserved_bw_data ** 2).sum(axis=1)
    n = reserved_bw_data.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        fairness_index = (sum_reserved_bw ** 2) / (n * sum_reserved_bw_sq)
        fairness_index = fairness_index.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    new_columns['fairness_index'] = fairness_index

    # Number of Saturated Links (>90% utilization)
    new_columns['saturated_links'] = (utilization > 0.9).sum(axis=1)

    # Maximum Utilization per Allocation Step
    new_columns['max_utilization'] = utilization.max(axis=1)

    # Throughput (assuming throughput is the total_reserved_bw)
    new_columns['throughput'] = new_columns['total_reserved_bw']

    # Assign all new columns at once to reduce fragmentation
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)

    metrics = {
        'total_reserved_bw': data['total_reserved_bw'],
        'total_available_bw': data['total_available_bw'],
        'average_utilization': data['average_utilization'],
        'fairness_index': data['fairness_index'],
        'saturated_links': data['saturated_links'],
        'max_utilization': data['max_utilization'],
        'throughput': data['throughput']
    }

    # Validation Checks
    for key, series in metrics.items():
        if series.isnull().all():
            print(f"Warning: Metric '{key}' contains all NaN values.")
        elif (series == 0).all():
            print(f"Warning: Metric '{key}' contains all zero values.")

    return metrics

def process_all_classes(categorized_files):
    """
    Process all categorized files to compute metrics per class.

    Args:
        categorized_files (dict): Nested dictionary with experiment sets and policies.

    Returns:
        dict: Nested dictionary with experiment sets, policies, and aggregated metrics.
    """
    class_metrics = {}
    for experiment_set, policies in categorized_files.items():
        class_metrics.setdefault(experiment_set, {})
        for policy, file_list in policies.items():
            print(f"\nProcessing experiment '{experiment_set}', policy '{policy}' with {len(file_list)} file(s).")
            combined_metrics = []
            for file in file_list:
                try:
                    data = pd.read_csv(file)

                    # Drop rows with any missing values to ensure data integrity
                    initial_row_count = data.shape[0]
                    data.dropna(inplace=True)
                    final_row_count = data.shape[0]
                    if final_row_count < initial_row_count:
                        print(f"Dropped {initial_row_count - final_row_count} incomplete rows from '{file}'.")

                    if data.empty:
                        print(f"Warning: File '{file}' is empty after dropping incomplete rows. Skipping.")
                        continue

                    metrics = compute_metrics(data)
                    combined_metrics.append(metrics)
                    print(f"Processed file '{file}' for experiment '{experiment_set}', policy '{policy}'.")
                except Exception as e:
                    print(f"Error processing file '{file}': {e}")

            if not combined_metrics:
                print(f"No valid data found for experiment '{experiment_set}', policy '{policy}'. Skipping.")
                continue

            # Aggregate metrics by averaging across files
            aggregated = {}
            for key in combined_metrics[0].keys():
                try:
                    # Concatenate all Series for this metric and compute the mean
                    concatenated = pd.concat([m[key] for m in combined_metrics], axis=1)
                    aggregated[key] = concatenated.mean(axis=1)
                except Exception as e:
                    print(f"Error aggregating metric '{key}' for experiment '{experiment_set}', policy '{policy}': {e}")

            class_metrics[experiment_set][policy] = aggregated
            print(f"Aggregated metrics for experiment '{experiment_set}', policy '{policy}'.")

    if not class_metrics:
        print("Error: No class metrics were computed. Please check your data and processing logic.")
        sys.exit(1)

    return class_metrics

def plot_metrics(class_metrics, save_dir='plots/bw'):
    """
    Plot various network metrics comparing different experiment sets and save the plots.

    Args:
        class_metrics (dict): Nested dictionary with experiment sets, policies, and aggregated metrics.
        save_dir (str): Directory path where plots will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    metrics_to_plot = {
        'Total Reserved Bandwidth': 'total_reserved_bw',
        'Total Available Bandwidth': 'total_available_bw',
        'Average Utilization': 'average_utilization',
        'Fairness Index (JFI)': 'fairness_index',
        'Number of Saturated Links (>90%)': 'saturated_links',
        'Maximum Utilization': 'max_utilization',
        'Throughput': 'throughput'
    }

    # Get list of all policies
    policies = set()
    for experiment_set in class_metrics:
        policies.update(class_metrics[experiment_set].keys())

    for policy in policies:
        for title, metric_key in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))
            has_data = False  # Flag to check if any plot was made
            for experiment_set in class_metrics:
                if policy in class_metrics[experiment_set]:
                    metrics = class_metrics[experiment_set][policy]
                    if metric_key in metrics and not metrics[metric_key].isnull().all():
                        series = metrics[metric_key]
                        plt.plot(series.index, series.values, label=experiment_set)
                        has_data = True
                    else:
                        print(f"No data for metric '{metric_key}' in experiment '{experiment_set}', policy '{policy}'. Skipping plot.")
            plt.title(f"{title} for Policy {policy}")
            plt.xlabel('Allocation Step')
            plt.ylabel(title)
            plt.grid(True)
            if has_data:
                plt.legend()
            else:
                print(f"No data plotted for '{title}' and policy '{policy}'. Legend will not be added.")
            plt.tight_layout()

            # Save the plot
            plot_filename = f"{metric_key}_{policy}.png".replace(' ', '_').replace('>', '').replace('(', '').replace(')', '').replace('/', '_')
            plot_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_path)
            print(f"Saved plot for '{title}' and policy '{policy}' as '{plot_path}'.")
            plt.close()

def plot_topology_statistics(topology, save_dir='plots/bw'):
    """
    (Optional) Plot network topology characteristics and save the plot.

    Args:
        topology (dict): Dictionary containing network topology parameters and their values.
        save_dir (str): Directory path where plots will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    categories = list(topology.keys())
    values = list(topology.values())

    plt.bar(categories, values, color='skyblue')
    plt.title('Network Topology Characteristics')
    plt.xlabel('Topology Parameters')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, 'network_topology_characteristics.png')
    plt.savefig(plot_path)
    print(f"Saved network topology plot as '{plot_path}'.")
    plt.close()

def main():
    """
    Main function to execute the script.
    """
    # Directory containing the CSV files
    csv_directory = '/home/fesposito/Andrea/tst/PlebyNet/'  # Replace with your actual path

    if not os.path.isdir(csv_directory):
        print(f"Error: The directory '{csv_directory}' does not exist.")
        sys.exit(1)

    # Network Topology Information (Optional: for additional plots)
    topology = {
        'Number of Spine Switches': 2,
        'Number of Leaf Switches': 5,
        'Hosts per Leaf': 10,
        'Max Spine Capacity (Gbps)': 500,
        'Max Leaf Capacity (Gbps)': 500,
        'Max Node Bandwidth (Gbps)': 100,
        'Max Leaf-to-Spine Bandwidth (Gbps)': 100
    }

    # Plot topology statistics (Optional)
    print("\nPlotting Network Topology Characteristics...")
    plot_topology_statistics(topology, save_dir='plots/bw')

    # Categorize CSV files
    print("\nCategorizing CSV files...")
    categorized_files = categorize_files(csv_directory)

    # Debug: Print categorized files
    print("\nCategorized Files:")
    for experiment_set, policies in categorized_files.items():
        print(f"\nExperiment Set: {experiment_set}")
        for policy, files in policies.items():
            print(f"  Policy: {policy}, Number of Files: {len(files)}")

    # Process each class to compute metrics
    print("\nProcessing classes to compute metrics...")
    class_metrics = process_all_classes(categorized_files)

    # Debug: Print summary of metrics
    print("\nSummary of Computed Metrics:")
    for experiment_set, policies in class_metrics.items():
        print(f"\nExperiment Set: {experiment_set}")
        for policy, metrics in policies.items():
            print(f"\nPolicy: {policy}")
            for metric_name, metric_series in metrics.items():
                print(f"\nMetric: {metric_name}")
                print(metric_series.describe())

    # Plot the computed metrics
    print("\nGenerating and Saving Plots...")
    plot_metrics(class_metrics, save_dir='plots/bw')
    print("\nPlotting Completed. All plots are saved in the 'plots/bw' directory.")

if __name__ == "__main__":
    main()
