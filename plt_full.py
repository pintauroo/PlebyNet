import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import os
from itertools import combinations, chain

def compute_confidence_interval(data, confidence=0.95):
    # [Existing implementation]
    n = len(data)
    if n < 2:
        return 0
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def remove_outliers_iqr(df, columns):
    # [Existing implementation]
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

def plot_confidence_intervals_by_utility(csv_file, label, confidence=0.95, output_dir='plots/confidence_intervals'):
    """
    Modified to include a 'label' parameter for distinguishing datasets.
    """
    # [Existing implementation with minor modifications]
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
        # Drop these rows
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
            # Select data for the current utility and column, dropping NaNs
            group_data = numeric_df[numeric_df['utility'] == utility][col].dropna()
            if group_data.empty:
                print(f"Warning: No valid data for utility '{utility}' in column '{col}'.")
                mean = np.nan
                ci = np.nan
            else:
                mean = group_data.mean()
                ci = compute_confidence_interval(group_data, confidence=confidence)
            summary_data[col][utility] = {'mean': mean, 'ci': ci}

    # Determine subplot layout (e.g., 2 columns per row)
    num_cols = len(selected_columns)
    n_cols_subplot = 2  # You can adjust this based on preference
    n_rows_subplot = math.ceil(num_cols / n_cols_subplot)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(n_cols_subplot * 7, n_rows_subplot * 6))
    axes = axes.flatten()  # Flatten in case of multiple rows

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    for idx, col in enumerate(selected_columns):
        ax = axes[idx]
        means = [summary_data[col][utility]['mean'] for utility in utility_types]
        cis = [summary_data[col][utility]['ci'] for utility in utility_types]
        x_pos = np.arange(num_utilities)

        # Handle cases where mean or CI might be NaN
        means_plot = [m if not math.isnan(m) else 0 for m in means]
        cis_plot = [c if not math.isnan(c) else 0 for c in cis]

        # Create bar plot with error bars
        bars = ax.bar(x_pos, means_plot, yerr=cis_plot, align='center', alpha=0.7, ecolor='black',
                      capsize=10, color=[utility_colors[utility] for utility in utility_types],
                      label=label)

        # Set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(utility_types, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'Confidence Intervals for {col} [{label}]')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate the mean values
        for bar, mean in zip(bars, means):
            if not math.isnan(mean):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height, f'{mean:.2f}', ha='center', va='bottom')

    # If there are unused subplots, remove them
    for idx in range(num_cols, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure with the label to distinguish different CSV files
    output_path = os.path.join(output_dir, f"{csv_file}_confidence_intervals_{label}.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Confidence intervals plot for '{label}' saved to '{output_path}'.")

def plot_cdf_by_utility(csv_file, label, t_gpu_min=None, t_gpu_max=None, output_dir='plots/cdf'):
    """
    Modified to include a 'label' parameter for distinguishing datasets.
    """
    # [Existing implementation with minor modifications]
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

    # Ensure 'utility' column exists
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return
    print(f"Initial number of rows: {len(df)}")

    # Apply t_gpu filters if specified
    if t_gpu_min is not None:
        df = df[df['t_gpu'] >= t_gpu_min]
    if t_gpu_max is not None:
        df = df[df['t_gpu'] < t_gpu_max]
    print(f"Number of rows after t_gpu filtering: {len(df)}")

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
    print("Utility counts after mapping:")
    print(df['utility'].value_counts())

    # Check for any unmapped utility types
    unmapped_utilities = df['utility'].isna()
    if unmapped_utilities.any():
        unique_unmapped = df.loc[unmapped_utilities, 'utility'].unique()
        print(f"Warning: Found unmapped utility types: {unique_unmapped}")
        # Drop these rows
        df = df.dropna(subset=['utility'])
        print(f"Dropped rows with unmapped utility types.")

    # Select the specified numerical columns
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned',
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]
    
    # Verify that the selected columns exist in the DataFrame
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

    # Define subplot layout
    num_cols = len(selected_columns)
    n_cols_subplot = 2  # Adjust as needed
    n_rows_subplot = math.ceil(num_cols / n_cols_subplot)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(n_cols_subplot * 7, n_rows_subplot * 6))
    axes = axes.flatten()  # Flatten in case of multiple rows

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    for idx, col in enumerate(selected_columns):
        ax = axes[idx]
        for utility in utility_types:
            # Extract data for the current utility and column
            data = numeric_df[numeric_df['utility'] == utility][col].dropna()
            if data.empty:
                print(f"Warning: No data for utility '{utility}' in column '{col}'. Skipping.")
                continue
            # Sort the data
            sorted_data = np.sort(data)
            # Compute the CDF values
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            # Plot the CDF
            ax.plot(sorted_data, cdf, label=utility, color=utility_colors[utility])

        # Set labels and title with label identifier
        ax.set_xlabel(col)
        ax.set_ylabel('CDF')
        ax.set_title(f'CDF of {col} by Utility [{label}]')
        ax.legend(title='Utility')
        ax.grid(True, linestyle='--', alpha=0.7)

    # If there are unused subplots, remove them
    for idx in range(num_cols, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure with the label to distinguish different CSV files
    output_path = os.path.join(output_dir, f"{csv_file}_cdf_{label}.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"CDF plot for '{label}' saved to '{output_path}'.")

# [Similarly, modify other plotting functions to accept 'label' and distinguish datasets]

if __name__ == "__main__":
    # List of CSV file paths without the '.csv' extension
    csv_files = [
        '50J_50N_NFD_HN_NDJ_SPS_NBW_test_results',
        '50J_50N_NFD_HN_NDJ_SPS_BW_test_results',
        '50J_50N_NFD_HN_NDJ_MPS_BW_test_results'
    ]

    # Define labels for each CSV file for identification in plots
    labels = [
        'SPS_NBW',
        'SPS_BW',
        'MPS_BW'
    ]

    # Ensure the number of labels matches the number of CSV files
    assert len(csv_files) == len(labels), "Number of labels must match number of CSV files."

    # Read the CSV files to compute percentiles (assuming percentiles are similar across datasets)
    # Alternatively, compute percentiles per CSV file if they differ
    # Here, we'll compute percentiles for each CSV separately
    t_gpu_ranges_list = []

    for csv_file in csv_files:
        try:
            df_main = pd.read_csv(csv_file + '.csv')
        except FileNotFoundError:
            print(f"Error: File '{csv_file}.csv' not found.")
            continue
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
            continue
        except pd.errors.ParserError:
            print("Error: CSV file is malformed.")
            continue

        # Check if 't_gpu' column exists
        if 't_gpu' not in df_main.columns:
            print(f"Error: 't_gpu' column not found in '{csv_file}.csv'.")
            continue

        # Drop NaN in 't_gpu'
        t_gpu_data = df_main['t_gpu'].dropna()

        if t_gpu_data.empty:
            print(f"Error: 't_gpu' column in '{csv_file}.csv' contains only NaN values.")
            continue

        # Compute percentiles
        q25 = t_gpu_data.quantile(0.25)
        q50 = t_gpu_data.quantile(0.50)
        q75 = t_gpu_data.quantile(0.75)
        max_t_gpu = t_gpu_data.max()
        min_t_gpu = t_gpu_data.min()

        print(f"Computed t_gpu Percentiles for '{csv_file}':")
        print(f"Min: {min_t_gpu}")
        print(f"25th Percentile: {q25}")
        print(f"50th Percentile (Median): {q50}")
        print(f"75th Percentile: {q75}")
        print(f"Max: {max_t_gpu}")

        # Define t_gpu_ranges based on percentiles
        t_gpu_ranges = [
            (min_t_gpu, q25),
            (q25, q50),
            (q50, q75),
            (q75, max_t_gpu)
        ]

        print(f"Defined t_gpu_ranges for '{csv_file}': {t_gpu_ranges}")

        t_gpu_ranges_list.append(t_gpu_ranges)

    # Define selected_columns and utility_types for subset plotting
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned',
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]

    # Define utility_types as per mapping
    utility_types = [
        'FRAG', 'SGF', 'LGF', 'SEQ', 'LIKELIHOOD', 'DRF', 'TETRIS'
    ]

    # Ensure the output directories exist
    os.makedirs('plots/confidence_intervals', exist_ok=True)
    os.makedirs('plots/cdf', exist_ok=True)
    os.makedirs('plots/cdf_custom_ranges', exist_ok=True)
    os.makedirs('plots/cdf_subsets', exist_ok=True)
    os.makedirs('plots/cdf_comparison', exist_ok=True)

    # Loop through each CSV file and its corresponding label
    for csv_file, label, t_gpu_ranges in zip(csv_files, labels, t_gpu_ranges_list):
        # Plot Confidence Intervals
        plot_confidence_intervals_by_utility(csv_file, label)

        # Plot CDFs without t_gpu filtering
        plot_cdf_by_utility(csv_file, label)

        # Plot Combined CDFs for custom t_gpu intervals and utilities
        # Assuming you have implemented 'plot_cdf_by_custom_t_gpu_ranges' to accept 'label'
        # If not, you should modify that function similarly to include 'label'
        # plot_cdf_by_custom_t_gpu_ranges(csv_file, t_gpu_ranges, label)

        # Explore all possible subsets of t_gpu ranges and plot CDFs
        # plot_cdf_by_all_subsets_of_t_gpu_ranges(
        #     csv_file=csv_file,
        #     t_gpu_ranges=t_gpu_ranges,
        #     selected_columns=selected_columns,
        #     utility_types=utility_types,
        #     output_dir='plots/cdf_subsets'
        # )

        # Optional: Compare CDFs across all intervals aggregated across utilities
        # Uncomment the following lines if you implement and want to use this function
        # plot_cdf_comparison_all_intervals(csv_file, t_gpu_ranges, selected_columns, utility_types, output_dir='plots/cdf_comparison')

    # Optional: Create combined comparison plots
    # Example: Compare Confidence Intervals across all CSV files for a specific numerical column
    # def plot_combined_confidence_intervals(csv_files, labels, selected_columns, output_dir='plots/combined_confidence_intervals'):
    #     os.makedirs(output_dir, exist_ok=True)
    #     for col in selected_columns:
    #         plt.figure(figsize=(10, 6))
    #         x_pos = np.arange(len(utility_types))
    #         width = 0.2  # Width of each bar
    #         for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
    #             # Extract data for the current column and label
    #             # You need to modify 'plot_confidence_intervals_by_utility' to return summary_data or compute here again
    #             # For simplicity, assume you have summary_data accessible
    #             # This requires further implementation
    #             pass  # Implement as needed
    #         plt.xlabel('Utility Types')
    #         plt.ylabel('Mean Value')
    #         plt.title(f'Combined Confidence Intervals for {col}')
    #         plt.legend(labels)
    #         plt.tight_layout()
    #         output_path = os.path.join(output_dir, f"combined_confidence_intervals_{col}.png")
    #         plt.savefig(output_path)
    #         plt.close()
    #         print(f"Combined confidence intervals plot for '{col}' saved to '{output_path}'.")

    # plot_combined_confidence_intervals(csv_files, labels, selected_columns)

    # Similarly, implement combined CDF comparison plots as needed

    print("All plots generated successfully.")
