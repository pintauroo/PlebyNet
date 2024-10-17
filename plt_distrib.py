import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math

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

def plot_confidence_intervals_by_utility(csv_file, confidence=0.95):
    """
    Read a CSV file, compute confidence intervals for selected numerical columns grouped by utility types,
    rename utility types, and plot them with each numerical column in its own subplot.

    Parameters:
        csv_file (str): Path to the CSV file.
        confidence (float): Confidence level for the intervals.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file+'.csv')
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
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
        'Utility.LIKELIHOOD': 'LIKELIHOOD'
    }

    # Rename utility types
    df['utility'] = df['utility'].map(utility_mapping)
    
    # Check for any unmapped utility types
    unmapped_utilities = df['utility'].isna()
    if unmapped_utilities.any():
        unique_unmapped = df.loc[unmapped_utilities, 'utility'].unique()
        print(f"Warning: Found unmapped utility types: {unique_unmapped}")
        # Optionally, you can drop these rows or handle them as needed
        df = df.dropna(subset=['utility'])
        print(f"Dropped rows with unmapped utility types.")

    # Select the specified numerical columns
    # selected_columns = ['first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned', 'jct', 'tot_unassigned']
    selected_columns = ['first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned', 'jct', 'tot_unassigned', 'discarded_jobs']

    
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
    plt.savefig(csv_file+'confidence_intervals.png')

def plot_cdf_by_utility(csv_file):
    """
    Read a CSV file, group data by utility types, and plot the Cumulative Distribution Function (CDF)
    for selected numerical columns.

    Parameters:
        csv_file (str): Path to the CSV file.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file+'.csv')
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
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
        'Utility.LIKELIHOOD': 'LIKELIHOOD'
    }

    # Rename utility types
    df['utility'] = df['utility'].map(utility_mapping)
    
    # Check for any unmapped utility types
    unmapped_utilities = df['utility'].isna()
    if unmapped_utilities.any():
        unique_unmapped = df.loc[unmapped_utilities, 'utility'].unique()
        print(f"Warning: Found unmapped utility types: {unique_unmapped}")
        # Optionally, you can drop these rows or handle them as needed
        df = df.dropna(subset=['utility'])
        print(f"Dropped rows with unmapped utility types.")

    # Select the specified numerical columns
    selected_columns = ['first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned', 'jct', 'tot_unassigned', 'discarded_jobs']
    
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

    # Define subplot layout
    num_cols = len(selected_columns)
    n_cols_subplot = 2  # Adjust as needed
    n_rows_subplot = math.ceil(num_cols / n_cols_subplot)

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
    plt.savefig(csv_file+'_cdf.png')

if __name__ == "__main__":
    # Replace 'final_allocations.csv' with your CSV file path
    csv_file_path = 'final_allocations'
    # If you need to use a different file, uncomment the next line
    # csv_file_path = 'fixed_duration_final_allocations'

    # Plot Confidence Intervals
    plot_confidence_intervals_by_utility(csv_file_path)

    # Plot CDFs
    plot_cdf_by_utility(csv_file_path)
