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

def categorize_files(directory, min_num=100, max_num=110):
    """
    Categorize CSV files based on predefined regex pattern and numeric range.

    Args:
        directory (str): Path to the directory containing CSV files.
        min_num (int): Minimum numeric prefix (inclusive).
        max_num (int): Maximum numeric prefix (inclusive).

    Returns:
        dict: Nested dictionary with experiment sets, policies, and lists of file paths.
    """
    categorized = {}
    all_files = glob.glob(os.path.join(directory, '*.csv'))

    if not all_files:
        print(f"No CSV files found in directory: {directory}")
        sys.exit(1)

    print(f"Found {len(all_files)} CSV file(s) in '{directory}'. Starting categorization...")

    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = file_pattern.match(filename)
        if match:
            # Extract the numeric prefix
            num_str = match.group(1)
            try:
                num = int(num_str)
            except ValueError:
                print(f"Warning: Unable to convert '{num_str}' to integer in file '{filename}'. Skipping.")
                continue

            # Check if the number is within the desired range
            if min_num <= num <= max_num:
                experiment_set = f"{match.group(2)}_{match.group(3)}"  # e.g., 'SPS_BW', 'MPS_BW', 'SPS_NBW'
                policy = match.group(4)  # 'TETRIS', 'DRF', etc.
                categorized.setdefault(experiment_set, {}).setdefault(policy, []).append(filepath)
                print(f"Categorized file '{filename}' under experiment '{experiment_set}', policy '{policy}'.")
            else:
                print(f"Skipping file '{filename}' with numeric prefix {num} outside the range {min_num}-{max_num}.")
        else:
            print(f"Warning: File '{filename}' did not match any known patterns and will be skipped.")

    if not categorized:
        print("Error: No files were categorized. Please check your regex patterns and file naming conventions.")
        sys.exit(1)

    return categorized

def compute_total_available_bandwidth(topology):
    """
    Compute the total available bandwidth from the network topology.

    Args:
        topology (dict): Dictionary containing network topology parameters.

    Returns:
        float: Total available bandwidth in Gbps.
    """
    num_spine_switches = topology.get('Number of Spine Switches', 0)
    num_leaf_switches = topology.get('Number of Leaf Switches', 0)
    hosts_per_leaf = topology.get('Hosts per Leaf', 0)
    max_leaf_to_spine_bw = topology.get('Max Leaf-to-Spine Bandwidth (Gbps)', 0)
    max_node_bw = topology.get('Max Node Bandwidth (Gbps)', 0)

    # Calculate total Leaf-Spine bandwidth
    total_leaf_spine_links = num_leaf_switches * num_spine_switches
    total_leaf_spine_bw = total_leaf_spine_links * max_leaf_to_spine_bw

    # Calculate total Host-Leaf bandwidth
    total_hosts = num_leaf_switches * hosts_per_leaf
    total_host_leaf_bw = total_hosts * max_node_bw

    # Total available bandwidth is sum of all link capacities
    total_available_bw = total_leaf_spine_bw + total_host_leaf_bw

    print(f"Total available bandwidth calculated from topology: {total_available_bw} Gbps")

    return total_available_bw

def compute_metrics(data, total_available_bw):
    """
    Compute various network metrics from the given DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing network bandwidth data.
        total_available_bw (float): Total available bandwidth from topology.

    Returns:
        dict: A dictionary containing computed metrics.
    """
    # Check if 'allocation_step' exists
    if 'allocation_step' not in data.columns:
        raise ValueError("Column 'allocation_step' not found in data.")

    # Convert 'allocation_step' to integer if it's not
    if not np.issubdtype(data['allocation_step'].dtype, np.number):
        data['allocation_step'] = pd.to_numeric(data['allocation_step'], errors='coerce')

    # Drop rows where 'allocation_step' could not be converted
    data.dropna(subset=['allocation_step'], inplace=True)
    data['allocation_step'] = data['allocation_step'].astype(int)

    # Set 'allocation_step' as the index
    data.set_index('allocation_step', inplace=True)

    # Optional: Include all allocation steps without filtering
    # If you still want to filter, uncomment the next line
    # data = data[(data.index >= 0) & (data.index <= 1000)]

    if data.empty:
        raise ValueError("Data is empty after filtering for allocation steps.")

    # Identify reserved bandwidth columns
    reserved_bw_columns = [col for col in data.columns if 'reserved_bw' in col]
    if not reserved_bw_columns:
        raise ValueError("No 'reserved_bw' columns found in data.")

    # Identify total bandwidth columns corresponding to reserved_bw
    total_bw_columns = [col.replace('reserved_bw', 'total_bw') for col in reserved_bw_columns]

    # Check if total_bw_columns exist
    missing_total_bw = [tb_col for tb_col in total_bw_columns if tb_col not in data.columns]
    if missing_total_bw:
        raise ValueError(f"Expected total bandwidth column(s) {missing_total_bw} not found in data.")

    # Identify available bandwidth columns
    available_bw_columns = [col for col in data.columns if 'available_bw' in col]
    if not available_bw_columns:
        raise ValueError("No 'available_bw' columns found in data.")

    # Convert bandwidth columns to numeric, coercing errors to NaN
    for col in reserved_bw_columns + total_bw_columns + available_bw_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Fill NaN values with zero after conversion
    data[reserved_bw_columns + total_bw_columns + available_bw_columns] = data[reserved_bw_columns + total_bw_columns + available_bw_columns].fillna(0.0)

    # **Divide all measured bandwidth variables by 100 to convert to Gbps**
    data[reserved_bw_columns + total_bw_columns + available_bw_columns] /= 100.0

    # Calculate required metrics
    # Prepare a dictionary to hold new columns
    new_columns = {}

    # Total Reserved Bandwidth
    new_columns['total_reserved_bw'] = data[reserved_bw_columns].sum(axis=1)

    # Normalize total_reserved_bw by total_available_bw
    new_columns['normalized_total_reserved_bw'] = new_columns['total_reserved_bw'] / total_available_bw

    # Total Available Bandwidth
    new_columns['total_available_bw'] = data[available_bw_columns].sum(axis=1)

    # Calculate utilization (reserved / total) for each link
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

    # Normalize throughput by total_available_bw
    new_columns['normalized_throughput'] = new_columns['throughput'] / total_available_bw

    # Assign all new columns at once to reduce fragmentation
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)

    metrics = {
        'total_reserved_bw': data['total_reserved_bw'],
        'normalized_total_reserved_bw': data['normalized_total_reserved_bw'],
        'total_available_bw': data['total_available_bw'],
        'average_utilization': data['average_utilization'],
        'fairness_index': data['fairness_index'],
        'saturated_links': data['saturated_links'],
        'max_utilization': data['max_utilization'],
        'throughput': data['throughput'],
        'normalized_throughput': data['normalized_throughput']
    }

    # Validation Checks
    for key, series in metrics.items():
        if series.isnull().all():
            print(f"Warning: Metric '{key}' contains all NaN values.")
        elif (series == 0).all():
            print(f"Warning: Metric '{key}' contains all zero values.")

    # Debug: Print first few rows of each metric
    print("\nComputed Metrics Sample:")
    for key, series in metrics.items():
        print(f"{key}:\n{series.head()}\n")

    return metrics

def process_all_classes(categorized_files, total_available_bw):
    """
    Process all categorized files to compute metrics per class.

    Args:
        categorized_files (dict): Nested dictionary with experiment sets and policies.
        total_available_bw (float): Total available bandwidth from topology.

    Returns:
        dict: Nested dictionary with experiment sets, policies, and aggregated metrics.
    """
    class_metrics = {}
    for experiment_set, policies in categorized_files.items():
        class_metrics.setdefault(experiment_set, {})
        for policy, file_list in policies.items():
            print(f"\nProcessing experiment '{experiment_set}', policy '{policy}' with {len(file_list)} file(s).")
            metric_dfs = []
            for file in file_list:
                try:
                    data = pd.read_csv(file)
                    
                    # **Divide all measured bandwidth variables by 100 to convert to Gbps**
                    # Identify bandwidth columns
                    reserved_bw_columns = [col for col in data.columns if 'reserved_bw' in col]
                    total_bw_columns = [col.replace('reserved_bw', 'total_bw') for col in reserved_bw_columns]
                    available_bw_columns = [col for col in data.columns if 'available_bw' in col]
                    bandwidth_columns = reserved_bw_columns + total_bw_columns + available_bw_columns
                    
                    # Convert to numeric and divide by 100
                    for col in bandwidth_columns:
                        if col in data.columns:
                            data[col] = pd.to_numeric(data[col], errors='coerce') / 100.0
                        else:
                            print(f"Warning: Expected column '{col}' not found in '{file}'. Filling with zeros.")
                            data[col] = 0.0

                    # Drop rows with any missing values to ensure data integrity
                    initial_row_count = data.shape[0]
                    data.dropna(inplace=True)
                    final_row_count = data.shape[0]
                    if final_row_count < initial_row_count:
                        print(f"Dropped {initial_row_count - final_row_count} incomplete rows from '{file}'.")

                    if data.empty:
                        print(f"Warning: File '{file}' is empty after dropping incomplete rows. Skipping.")
                        continue

                    metrics = compute_metrics(data, total_available_bw)
                    
                    # Convert metrics dict to DataFrame, reset index to have 'allocation_step'
                    metrics_df = pd.DataFrame(metrics).reset_index()
                    metric_dfs.append(metrics_df)

                    print(f"Processed file '{file}' for experiment '{experiment_set}', policy '{policy}'. Metrics Sample:")
                    print(metrics_df.head())
                except Exception as e:
                    print(f"Error processing file '{file}': {e}")

            if not metric_dfs:
                print(f"No valid data found for experiment '{experiment_set}', policy '{policy}'. Skipping.")
                continue

            # Concatenate all metric DataFrames
            concatenated_df = pd.concat(metric_dfs, ignore_index=True)

            # Optional: Include all allocation steps without filtering
            # If you still want to filter, uncomment the next line
            # concatenated_df = concatenated_df[(concatenated_df['allocation_step'] >= 0) & (concatenated_df['allocation_step'] <= 1000)]

            if concatenated_df.empty:
                print(f"No data in the allocation step range for experiment '{experiment_set}', policy '{policy}'. Skipping.")
                continue

            # Group by 'allocation_step' and compute mean
            aggregated = concatenated_df.groupby('allocation_step').mean()

            if aggregated.empty:
                print(f"No aggregated data for experiment '{experiment_set}', policy '{policy}'. Skipping.")
                continue

            # Debug: Print aggregated metrics
            print(f"Aggregated metrics for experiment '{experiment_set}', policy '{policy}'. Sample:")
            print(aggregated.head())

            class_metrics[experiment_set][policy] = aggregated.to_dict(orient='series')

    if not class_metrics:
        print("Error: No class metrics were computed. Please check your data and processing logic.")
        sys.exit(1)

    return class_metrics

def plot_metrics(class_metrics, save_dir='plots/bw'):
    """
    Plot various network metrics comparing different utility functions together for each metric.

    Args:
        class_metrics (dict): Nested dictionary with experiment sets, policies, and aggregated metrics.
        save_dir (str): Directory path where plots will be saved.
    """
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import os

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the utility function metrics you want to plot
    utility_metrics_to_plot = {
        'Total Reserved Bandwidth (Normalized)': 'normalized_total_reserved_bw',
        'Total Available Bandwidth': 'total_available_bw',
        'Average Utilization': 'average_utilization',
        'Fairness Index (JFI)': 'fairness_index',
        'Number of Saturated Links (>90%)': 'saturated_links',
        'Maximum Utilization': 'max_utilization',
        'Throughput (Normalized)': 'normalized_throughput'
    }

    # Extract all unique policies (assuming policies represent utility functions)
    policies = set()
    for experiment_set in class_metrics:
        policies.update(class_metrics[experiment_set].keys())

    # Iterate over each metric to create a consolidated plot
    for metric_title, metric_key in utility_metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        has_data = False  # Flag to check if any plot was made for the metric

        for policy in policies:
            # Collect metric data across all experiment sets for the current policy
            metric_series_list = []
            for experiment_set in class_metrics:
                if policy in class_metrics[experiment_set]:
                    metrics = class_metrics[experiment_set][policy]
                    series = pd.Series(metrics[metric_key])
                    if not series.empty:
                        metric_series_list.append(series)

            if not metric_series_list:
                print(f"No data found for policy '{policy}' and metric '{metric_key}'. Skipping.")
                continue

            # Concatenate all series and compute the mean across experiment sets
            combined_series = pd.concat(metric_series_list, axis=1)
            averaged_series = combined_series.mean(axis=1)

            # Ensure the series is sorted by allocation_step
            averaged_series = averaged_series.sort_index()

            if averaged_series.empty:
                print(f"Averaged series for policy '{policy}' and metric '{metric_key}' is empty. Skipping.")
                continue

            # Apply LOWESS smoothing for a smoother curve
            smoothed = lowess(averaged_series, averaged_series.index, frac=0.1, return_sorted=False)

            # Plot the smoothed series
            plt.plot(averaged_series.index, smoothed, label=policy)
            has_data = True
            print(f"Plotted metric '{metric_key}' for policy '{policy}' with LOWESS smoothing.")

        if not has_data:
            print(f"No data available to plot for metric '{metric_title}'. Skipping plot.")
            plt.close()
            continue

        # Customize the plot
        plt.title(f"{metric_title} Comparison Across Utility Functions")
        plt.xlabel('Allocation Step')
        plt.ylabel(metric_title)
        plt.grid(True)
        plt.legend(title='Utility Functions')
        plt.tight_layout()
        plt.xlim(left=0)  # Start x-axis from 0
        plt.ylim(bottom=0)  # Ensure y-axis starts from 0
        plt.yscale('linear')  # Use linear scale since no negative values

        # Prepare a safe filename by removing or replacing problematic characters
        safe_metric_key = metric_key.replace(' ', '_').replace('>', '').replace('(', '').replace(')', '').replace('/', '_')

        # Save the plot as PNG
        plot_filename_png = f"{safe_metric_key}_utility_comparison.png"
        plot_path_png = os.path.join(save_dir, plot_filename_png)
        plt.savefig(plot_path_png)
        print(f"Saved plot for '{metric_title}' as '{plot_path_png}'.")

        # Save the plot as SVG
        plot_filename_svg = f"{safe_metric_key}_utility_comparison.svg"
        plot_path_svg = os.path.join(save_dir, plot_filename_svg)
        plt.savefig(plot_path_svg)
        print(f"Saved plot for '{metric_title}' as '{plot_path_svg}'.")

        plt.close()

    print("\nAll comparison plots have been generated and saved in the 'plots/bw' directory.")

def plot_topology_statistics(topology, save_dir='plots/bw'):
    """
    (Optional) Plot network topology characteristics and save the plot.

    Args:
        topology (dict): Dictionary containing network topology parameters and their values.
        save_dir (str): Directory path where plots will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    categories = list(topology.keys())
    values = list(topology.values())

    plt.bar(categories, values, color='skyblue')
    plt.title('Network Topology Characteristics')
    plt.xlabel('Topology Parameters')
    plt.ylabel('Values (Gbps and Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Prepare a safe filename by removing or replacing problematic characters
    safe_filename_base = 'network_topology_characteristics'

    # Save the plot as PNG
    plot_path_png = os.path.join(save_dir, f"{safe_filename_base}.png")
    plt.savefig(plot_path_png)
    print(f"Saved network topology plot as '{plot_path_png}'.")

    # Save the plot as SVG
    plot_path_svg = os.path.join(save_dir, f"{safe_filename_base}.svg")
    plt.savefig(plot_path_svg)
    print(f"Saved network topology plot as '{plot_path_svg}'.")

    plt.close()

def plot_aggregated_metrics(class_metrics, utility_metrics_to_plot, save_dir='plots/bw'):
    """
    (Optional) Plot aggregated metrics as bar charts comparing policies and experiment sets.

    Args:
        class_metrics (dict): Nested dictionary with experiment sets, policies, and aggregated metrics.
        utility_metrics_to_plot (dict): Dictionary mapping metric titles to metric keys.
        save_dir (str): Directory path where plots will be saved.
    """
    import matplotlib.pyplot as plt
    import os

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Compute average metric values over allocation steps for each policy and experiment set
    # Prepare data for plotting aggregated metrics per policy
    aggregated_metrics_per_policy = {}
    for metric_title, metric_key in utility_metrics_to_plot.items():
        aggregated_metrics_per_policy[metric_title] = {}
        for policy in set(policy for policies in class_metrics.values() for policy in policies):
            metric_values = []
            for experiment_set in class_metrics:
                if policy in class_metrics[experiment_set]:
                    metrics = class_metrics[experiment_set][policy]
                    series = pd.Series(metrics[metric_key])
                    if not series.empty:
                        mean_value = series.mean()
                        metric_values.append(mean_value)
            if metric_values:
                aggregated_metrics_per_policy[metric_title][policy] = np.mean(metric_values)
            else:
                aggregated_metrics_per_policy[metric_title][policy] = np.nan

    # Plot aggregated metrics per policy
    for metric_title, metric_data in aggregated_metrics_per_policy.items():
        plt.figure(figsize=(12, 8))
        policies_list = list(metric_data.keys())
        values = [metric_data[policy] for policy in policies_list]
        plt.bar(policies_list, values, color='skyblue')
        plt.title(f"Average {metric_title} per Policy (Allocation Steps 0-1000)")
        plt.xlabel('Policy')
        plt.ylabel(f"Average {metric_title}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.ylim(bottom=0)  # Ensure y-axis starts from 0
        plt.grid(True, axis='y')

        # Save the plot
        safe_metric_title = metric_title.replace(' ', '_').replace('>', '').replace('(', '').replace(')', '').replace('/', '_')
        plot_filename_png = f"Average_{safe_metric_title}_per_Policy.png"
        plot_path_png = os.path.join(save_dir, plot_filename_png)
        plt.savefig(plot_path_png)
        print(f"Saved aggregated metric plot '{plot_path_png}'.")

        # Save the plot as SVG
        plot_filename_svg = f"Average_{safe_metric_title}_per_Policy.svg"
        plot_path_svg = os.path.join(save_dir, plot_filename_svg)
        plt.savefig(plot_path_svg)
        print(f"Saved aggregated metric plot '{plot_path_svg}'.")

        plt.close()

    # Prepare data for plotting aggregated metrics per experiment set
    aggregated_metrics_per_experiment_set = {}
    for metric_title, metric_key in utility_metrics_to_plot.items():
        aggregated_metrics_per_experiment_set[metric_title] = {}
        for experiment_set in class_metrics:
            metric_values = []
            for policy in class_metrics[experiment_set]:
                metrics = class_metrics[experiment_set][policy]
                series = pd.Series(metrics[metric_key])
                if not series.empty:
                    mean_value = series.mean()
                    metric_values.append(mean_value)
            if metric_values:
                aggregated_metrics_per_experiment_set[metric_title][experiment_set] = np.mean(metric_values)
            else:
                aggregated_metrics_per_experiment_set[metric_title][experiment_set] = np.nan

    # Plot aggregated metrics per experiment set
    for metric_title, metric_data in aggregated_metrics_per_experiment_set.items():
        plt.figure(figsize=(12, 8))
        experiment_set_list = list(metric_data.keys())
        values = [metric_data[experiment_set] for experiment_set in experiment_set_list]
        plt.bar(experiment_set_list, values, color='skyblue')
        plt.title(f"Average {metric_title} per Experiment Set (Allocation Steps 0-1000)")
        plt.xlabel('Experiment Set')
        plt.ylabel(f"Average {metric_title}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.ylim(bottom=0)  # Ensure y-axis starts from 0
        plt.grid(True, axis='y')

        # Save the plot
        safe_metric_title = metric_title.replace(' ', '_').replace('>', '').replace('(', '').replace(')', '').replace('/', '_')
        plot_filename_png = f"Average_{safe_metric_title}_per_Experiment_Set.png"
        plot_path_png = os.path.join(save_dir, plot_filename_png)
        plt.savefig(plot_path_png)
        print(f"Saved aggregated metric plot '{plot_path_png}'.")

        # Save the plot as SVG
        plot_filename_svg = f"Average_{safe_metric_title}_per_Experiment_Set.svg"
        plot_path_svg = os.path.join(save_dir, plot_filename_svg)
        plt.savefig(plot_path_svg)
        print(f"Saved aggregated metric plot '{plot_path_svg}'.")

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

    plt.figure(figsize=(12, 8))
    categories = list(topology.keys())
    values = list(topology.values())

    plt.bar(categories, values, color='skyblue')
    plt.title('Network Topology Characteristics')
    plt.xlabel('Topology Parameters')
    plt.ylabel('Values (Gbps and Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Prepare a safe filename by removing or replacing problematic characters
    safe_filename_base = 'network_topology_characteristics'

    # Save the plot as PNG
    plot_path_png = os.path.join(save_dir, f"{safe_filename_base}.png")
    plt.savefig(plot_path_png)
    print(f"Saved network topology plot as '{plot_path_png}'.")

    # Save the plot as SVG
    plot_path_svg = os.path.join(save_dir, f"{safe_filename_base}.svg")
    plt.savefig(plot_path_svg)
    print(f"Saved network topology plot as '{plot_path_svg}'.")

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

    # Compute total available bandwidth from topology
    total_available_bw = compute_total_available_bandwidth(topology)

    # Plot topology statistics (Optional)
    print("\nPlotting Network Topology Characteristics...")
    plot_topology_statistics(topology, save_dir='plots/bw')

    # Categorize CSV files with the desired numeric range
    print("\nCategorizing CSV files...")
    categorized_files = categorize_files(csv_directory, min_num=100, max_num=110)

    # Debug: Print categorized files
    print("\nCategorized Files:")
    for experiment_set, policies in categorized_files.items():
        print(f"\nExperiment Set: {experiment_set}")
        for policy, files in policies.items():
            print(f"  Policy: {policy}, Number of Files: {len(files)}")
            for f in files:
                print(f"    - {os.path.basename(f)}")

    # Process each class to compute metrics
    print("\nProcessing classes to compute metrics...")
    class_metrics = process_all_classes(categorized_files, total_available_bw)

    # Debug: Print summary of metrics
    print("\nSummary of Computed Metrics:")
    for experiment_set, policies in class_metrics.items():
        print(f"\nExperiment Set: {experiment_set}")
        for policy, metrics in policies.items():
            print(f"\nPolicy: {policy}")
            for metric_name, metric_series in metrics.items():
                print(f"\nMetric: {metric_name}")
                print(pd.Series(metric_series).describe())

    # Plot the computed metrics
    print("\nGenerating and Saving Comparison Plots...")
    plot_metrics(class_metrics, save_dir='plots/bw')
    print("\nPlotting Completed. All comparison plots are saved in the 'plots/bw' directory.")

if __name__ == "__main__":
    main()
