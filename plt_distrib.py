import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import os
from itertools import combinations

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for a list of numbers.

    Parameters:
        data (array-like): The data points.
        confidence (float): The confidence level (default is 0.95 for 95% confidence).

    Returns:
        float: The margin of error.
    """
    n = len(data)
    if n < 2:
        return 0  # Not enough data to compute confidence interval
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of the mean
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)  # Margin of error
    return h

def plot_confidence_intervals_by_utility(csv_file, confidence=0.95, output_dir='plots/confidence_intervals'):
    """
    Read a CSV file, compute confidence intervals for selected numerical columns grouped by utility types,
    rename utility types, and plot them with each numerical column in its own subplot.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        confidence (float): Confidence level for the intervals.
        output_dir (str): Directory to save the plots.
    """
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
                      capsize=10, color=[utility_colors[utility] for utility in utility_types])

        # Set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(utility_types, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'Confidence Intervals for {col}')
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

    # Save the figure
    output_path = os.path.join(output_dir, f"{csv_file}_confidence_intervals.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Confidence intervals plot saved to '{output_path}'.")

def plot_cdf_by_utility(csv_file, t_gpu_min=None, t_gpu_max=None, output_dir='plots/cdf'):
    """
    Read a CSV file, group data by utility types, and plot the Cumulative Distribution Function (CDF)
    for selected numerical columns.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        t_gpu_min (int, optional): Minimum t_gpu value to filter the data.
        t_gpu_max (int, optional): Maximum t_gpu value to filter the data.
        output_dir (str): Directory to save the plots.
    """
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

        # Set labels and title
        ax.set_xlabel(col)
        ax.set_ylabel('CDF')
        ax.set_title(f'CDF of {col} by Utility')
        ax.legend(title='Utility')
        ax.grid(True, linestyle='--', alpha=0.7)

    # If there are unused subplots, remove them
    for idx in range(num_cols, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, f"{csv_file}_cdf.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"CDF plot saved to '{output_path}'.")

def plot_cdf_by_all_subsets_of_t_gpu_ranges(csv_file, t_gpu_ranges, selected_columns, utility_types, output_dir='plots/cdf_subsets'):
    """
    Explore all possible subsets of t_gpu ranges and plot the CDFs for each subset, grouped by utility types.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        t_gpu_ranges (list of tuples): List of (t_gpu_min, t_gpu_max) intervals.
        selected_columns (list): List of numerical columns to plot.
        utility_types (list): List of utility types.
        output_dir (str): Directory to save the plots.
    """
    from itertools import chain, combinations

    def all_subsets(ranges):
        """Generate all non-empty subsets of the given ranges."""
        return chain.from_iterable(combinations(ranges, r) for r in range(1, len(ranges)+1))

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

    # Ensure 'utility' and 't_gpu' columns exist
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return
    if 't_gpu' not in df.columns:
        print("Error: 't_gpu' column not found in the CSV file.")
        return
    print(f"Initial number of rows: {len(df)}")

    # Drop rows where 'utility' or 't_gpu' is NaN
    initial_row_count = len(df)
    df = df.dropna(subset=['utility', 't_gpu'])
    final_row_count = len(df)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN in 'utility' or 't_gpu' columns.")

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

    # Verify that the selected columns exist in the dataframe
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing in the CSV file: {missing_columns}")
        return

    numeric_df = df[selected_columns + ['utility', 't_gpu']].copy()

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate all non-empty subsets of t_gpu_ranges
    subsets = list(all_subsets(t_gpu_ranges))
    print(f"Total number of non-empty subsets: {len(subsets)}")

    for subset in subsets:
        # Create a label for the subset
        subset_label = "_".join([f"{min}_{max}" for (min, max) in subset])
        
        # Filter DataFrame to include data within any of the ranges in the subset
        df_subset = numeric_df.copy()
        condition = False
        for (t_min, t_max) in subset:
            condition = condition | ((df_subset['t_gpu'] >= t_min) & (df_subset['t_gpu'] < t_max))
        df_subset = df_subset[condition]
        
        if df_subset.empty:
            print(f"Warning: No data for subset {subset_label}. Skipping.")
            continue

        for col in selected_columns:
            plt.figure(figsize=(10, 6))
            for utility in utility_types:
                # Extract data for the current utility and column
                data = df_subset[df_subset['utility'] == utility][col].dropna()
                if data.empty:
                    print(f"Warning: No data for utility '{utility}' in subset '{subset_label}' for column '{col}'. Skipping.")
                    continue
                # Sort the data
                sorted_data = np.sort(data)
                # Compute the CDF values
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                # Plot the CDF
                plt.plot(sorted_data, cdf, label=utility, color=utility_colors[utility])

            plt.xlabel(col)
            plt.ylabel('CDF')
            plt.title(f'CDF of {col} for t_gpu Ranges: {subset_label}')
            plt.legend(title='Utility', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(output_dir, f"{csv_file}_cdf_subset_{subset_label}_{col}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"CDF plot for subset '{subset_label}' and column '{col}' saved to '{output_path}'.")

def plot_cdf_by_custom_t_gpu_ranges(csv_file, t_gpu_ranges, output_dir='plots/cdf_custom_ranges'):
    """
    For each selected numerical column, create a figure with subplots for each custom t_gpu interval.
    In each subplot, plot the CDFs of different utilities.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        t_gpu_ranges (list of tuples): List of (t_gpu_min, t_gpu_max) intervals.
        output_dir (str): Directory to save the plots.
    """
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

    # Ensure 'utility' and 't_gpu' columns exist
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return
    if 't_gpu' not in df.columns:
        print("Error: 't_gpu' column not found in the CSV file.")
        return
    print(f"Initial number of rows: {len(df)}")

    # Drop rows where 'utility' or 't_gpu' is NaN
    initial_row_count = len(df)
    df = df.dropna(subset=['utility', 't_gpu'])
    final_row_count = len(df)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN in 'utility' or 't_gpu' columns.")

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

    numeric_df = df[selected_columns + ['utility', 't_gpu']].copy()

    # Get unique utility types after renaming
    utility_types = numeric_df['utility'].unique()
    print(f"Utility Types after renaming: {utility_types}")
    num_utilities = len(utility_types)

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select the numerical columns
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned',
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]

    # Define subplot layout based on the number of t_gpu_ranges
    n_intervals = len(t_gpu_ranges)
    n_cols_subplot = 2  # You can adjust this based on preference
    n_rows_subplot = math.ceil(n_intervals / n_cols_subplot)

    # Create subplots for each numerical column
    for col in selected_columns:
        fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(n_cols_subplot * 7, n_rows_subplot * 6))
        axes = axes.flatten()  # Flatten in case of multiple rows

        for idx, (t_gpu_min, t_gpu_max) in enumerate(t_gpu_ranges):
            ax = axes[idx]
            # Filter DataFrame to the t_gpu interval
            df_interval = numeric_df[(numeric_df['t_gpu'] >= t_gpu_min) & (numeric_df['t_gpu'] < t_gpu_max)]
            if df_interval.empty:
                print(f"Warning: No data for t_gpu interval ({t_gpu_min}, {t_gpu_max}) in column '{col}'. Skipping.")
                continue
            for utility in utility_types:
                # Extract data for the current utility and column
                data = df_interval[df_interval['utility'] == utility][col].dropna()
                if data.empty:
                    print(f"Warning: No data for utility '{utility}' in t_gpu interval ({t_gpu_min}, {t_gpu_max}) for column '{col}'. Skipping.")
                    continue
                # Sort the data
                sorted_data = np.sort(data)
                # Compute the CDF values
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                # Plot the CDF
                ax.plot(sorted_data, cdf, label=utility, color=utility_colors[utility])

            # Set labels and title
            ax.set_xlabel(col)
            ax.set_ylabel('CDF')
            ax.set_title(f't_gpu [{t_gpu_min}, {t_gpu_max})')
            ax.legend(title='Utility', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)

        # Remove unused subplots
        for idx in range(n_intervals, len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f"{csv_file}_cdf_custom_{col}.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"CDF custom plot for column '{col}' saved to '{output_path}'.")

def plot_cdf_by_all_subsets_of_t_gpu_ranges(csv_file, t_gpu_ranges, selected_columns, utility_types, output_dir='plots/cdf_subsets'):
    """
    Explore all possible subsets of t_gpu ranges and plot the CDFs for each subset, grouped by utility types.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        t_gpu_ranges (list of tuples): List of (t_gpu_min, t_gpu_max) intervals.
        selected_columns (list): List of numerical columns to plot.
        utility_types (list): List of utility types.
        output_dir (str): Directory to save the plots.
    """
    from itertools import chain, combinations

    def all_subsets(ranges):
        """Generate all non-empty subsets of the given ranges."""
        return chain.from_iterable(combinations(ranges, r) for r in range(1, len(ranges)+1))

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

    # Ensure 'utility' and 't_gpu' columns exist
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return
    if 't_gpu' not in df.columns:
        print("Error: 't_gpu' column not found in the CSV file.")
        return
    print(f"Initial number of rows: {len(df)}")

    # Drop rows where 'utility' or 't_gpu' is NaN
    initial_row_count = len(df)
    df = df.dropna(subset=['utility', 't_gpu'])
    final_row_count = len(df)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN in 'utility' or 't_gpu' columns.")

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

    # Verify that the selected columns exist in the dataframe
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing in the CSV file: {missing_columns}")
        return

    numeric_df = df[selected_columns + ['utility', 't_gpu']].copy()

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate all non-empty subsets of t_gpu_ranges
    subsets = list(all_subsets(t_gpu_ranges))
    print(f"Total number of non-empty subsets: {len(subsets)}")

    for subset in subsets:
        # Create a label for the subset
        subset_label = "_".join([f"{min}_{max}" for (min, max) in subset])
        
        # Filter DataFrame to include data within any of the ranges in the subset
        df_subset = numeric_df.copy()
        condition = False
        for (t_min, t_max) in subset:
            condition = condition | ((df_subset['t_gpu'] >= t_min) & (df_subset['t_gpu'] < t_max))
        df_subset = df_subset[condition]
        
        if df_subset.empty:
            print(f"Warning: No data for subset {subset_label}. Skipping.")
            continue

        for col in selected_columns:
            plt.figure(figsize=(10, 6))
            for utility in utility_types:
                # Extract data for the current utility and column
                data = df_subset[df_subset['utility'] == utility][col].dropna()
                if data.empty:
                    print(f"Warning: No data for utility '{utility}' in subset '{subset_label}' for column '{col}'. Skipping.")
                    continue
                # Sort the data
                sorted_data = np.sort(data)
                # Compute the CDF values
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                # Plot the CDF
                plt.plot(sorted_data, cdf, label=utility, color=utility_colors[utility])

            plt.xlabel(col)
            plt.ylabel('CDF')
            plt.title(f'CDF of {col} for t_gpu Ranges: {subset_label}')
            plt.legend(title='Utility', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(output_dir, f"{csv_file}_cdf_subset_{subset_label}_{col}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"CDF plot for subset '{subset_label}' and column '{col}' saved to '{output_path}'.")

def plot_cdf_by_custom_t_gpu_ranges(csv_file, t_gpu_ranges, output_dir='plots/cdf_custom_ranges'):
    """
    For each selected numerical column, create a figure with subplots for each custom t_gpu interval.
    In each subplot, plot the CDFs of different utilities.

    Parameters:
        csv_file (str): Path to the CSV file without the '.csv' extension.
        t_gpu_ranges (list of tuples): List of (t_gpu_min, t_gpu_max) intervals.
        output_dir (str): Directory to save the plots.
    """
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

    # Ensure 'utility' and 't_gpu' columns exist
    if 'utility' not in df.columns:
        print("Error: 'utility' column not found in the CSV file.")
        return
    if 't_gpu' not in df.columns:
        print("Error: 't_gpu' column not found in the CSV file.")
        return
    print(f"Initial number of rows: {len(df)}")

    # Drop rows where 'utility' or 't_gpu' is NaN
    initial_row_count = len(df)
    df = df.dropna(subset=['utility', 't_gpu'])
    final_row_count = len(df)
    dropped_rows = initial_row_count - final_row_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN in 'utility' or 't_gpu' columns.")

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

    numeric_df = df[selected_columns + ['utility', 't_gpu']].copy()

    # Get unique utility types after renaming
    utility_types = numeric_df['utility'].unique()
    print(f"Utility Types after renaming: {utility_types}")
    num_utilities = len(utility_types)

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types)}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select the numerical columns
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned',
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]

    # Define subplot layout based on the number of t_gpu_ranges
    n_intervals = len(t_gpu_ranges)
    n_cols_subplot = 2  # You can adjust this based on preference
    n_rows_subplot = math.ceil(n_intervals / n_cols_subplot)

    # Create subplots for each numerical column
    for col in selected_columns:
        fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(n_cols_subplot * 7, n_rows_subplot * 6))
        axes = axes.flatten()  # Flatten in case of multiple rows

        for idx, (t_gpu_min, t_gpu_max) in enumerate(t_gpu_ranges):
            ax = axes[idx]
            # Filter DataFrame to the t_gpu interval
            df_interval = numeric_df[(numeric_df['t_gpu'] >= t_gpu_min) & (numeric_df['t_gpu'] < t_gpu_max)]
            if df_interval.empty:
                print(f"Warning: No data for t_gpu interval ({t_gpu_min}, {t_gpu_max}) in column '{col}'. Skipping.")
                continue
            for utility in utility_types:
                # Extract data for the current utility and column
                data = df_interval[df_interval['utility'] == utility][col].dropna()
                if data.empty:
                    print(f"Warning: No data for utility '{utility}' in t_gpu interval ({t_gpu_min}, {t_gpu_max}) for column '{col}'. Skipping.")
                    continue
                # Sort the data
                sorted_data = np.sort(data)
                # Compute the CDF values
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                # Plot the CDF
                ax.plot(sorted_data, cdf, label=utility, color=utility_colors[utility])

            # Set labels and title
            ax.set_xlabel(col)
            ax.set_ylabel('CDF')
            ax.set_title(f't_gpu [{t_gpu_min}, {t_gpu_max})')
            ax.legend(title='Utility', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)

        # Remove unused subplots
        for idx in range(n_intervals, len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f"{csv_file}_cdf_custom_{col}.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"CDF custom plot for column '{col}' saved to '{output_path}'.")

if __name__ == "__main__":
    # Replace 'final_allocations.csv' with your actual CSV file path without the '.csv' extension
    csv_file_path = 'final_allocations'  # Example: 'final_allocations'

    # Define the t_gpu intervals you want to compare
    # You can define custom significant ranges here
    # Example 1: Based on Quantiles (commented out)
    # t_gpu_ranges = [
    #     (700, 850),  # 0-25th percentile
    #     (850, 950),  # 25th-50th percentile
    #     (950, 1050), # 50th-75th percentile
    #     (1050, 1200) # 75th-100th percentile
    # ]

    # Example 2: Based on Natural Breaks or Domain Knowledge
    t_gpu_ranges = [
        (700, 900),
        (900, 1000),
        (1000, 1200)
    ]

    # Example 3: Multiple overlapping or custom ranges
    # t_gpu_ranges = [
    #     (700, 800),
    #     (750, 850),
    #     (800, 900),
    #     (850, 950),
    #     (900, 1000),
    #     (950, 1050),
    #     (1000, 1100),
    #     (1050, 1200)
    # ]

    # Ensure the output directories exist
    os.makedirs('plots/confidence_intervals', exist_ok=True)
    os.makedirs('plots/cdf', exist_ok=True)
    os.makedirs('plots/cdf_custom_ranges', exist_ok=True)
    os.makedirs('plots/cdf_subsets', exist_ok=True)
    os.makedirs('plots/cdf_comparison', exist_ok=True)

    # Plot Confidence Intervals
    plot_confidence_intervals_by_utility(csv_file_path)

    # Plot CDFs without t_gpu filtering
    plot_cdf_by_utility(csv_file_path)

    # Plot Combined CDFs for custom t_gpu intervals and utilities
    plot_cdf_by_custom_t_gpu_ranges(csv_file_path, t_gpu_ranges)

    # Explore all possible subsets of t_gpu ranges and plot CDFs
    # Define selected_columns and utility_types for subset plotting
    # These should match the ones used in other functions
    selected_columns = [
        'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned',
        'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
    ]

    # Define utility_types as per mapping
    utility_types = [
        'FRAG', 'SGF', 'LGF', 'SEQ', 'LIKELIHOOD', 'DRF', 'TETRIS'
    ]

    plot_cdf_by_all_subsets_of_t_gpu_ranges(
        csv_file=csv_file_path,
        t_gpu_ranges=t_gpu_ranges,
        selected_columns=selected_columns,
        utility_types=utility_types,
        output_dir='plots/cdf_subsets'
    )

    # Optional: Compare CDFs across all intervals aggregated across utilities
    # Uncomment the following line if you want to generate this plot
    # plot_cdf_comparison_all_intervals(csv_file_path, t_gpu_ranges, selected_columns, utility_types, output_dir='plots/cdf_comparison')
