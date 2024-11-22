import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import os
import tikzplotlib

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

def set_y_label(col):
    if col == 'first_unassigned':    
        ylabel = 'First Failure'
    elif col == 'tot_unassigned':
        ylabel = 'Allocation Failure Rate'
    elif col == 'jct_mean':
        ylabel = 'JCT'
    elif col == 'jct_median':
        ylabel = 'JCT Median'
    else:
        ylabel = col
    return ylabel

def plot_confidence_intervals_by_utility(csv_file, label, confidence=0.95, t_gpu_min=None, t_gpu_max=None, 
                                         output_dir='plots/confidence_intervals', text_size=25, remove_outliers=True):
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

    # Remove outliers if flag is set
    if remove_outliers:
        df = remove_outliers_iqr(df, selected_columns)

    numeric_df = df[selected_columns].copy()

    # Convert tot_unassigned to percentage
    numeric_df['tot_unassigned'] = (numeric_df['tot_unassigned'] / 70) * 100

    # Combine the selected numerical columns with the 'utility' column
    numeric_df['utility'] = df['utility']

    # Get unique utility types after renaming
    utility_types = numeric_df['utility'].unique()
    print(f"Utility Types after renaming: {utility_types}")
    utility_types_sorted = sorted(utility_types)
    num_utilities = len(utility_types_sorted)

    # Compute mean and confidence intervals for each numerical column grouped by utility
    summary_data = {}
    for col in selected_columns:
        summary_data[col] = {}
        for utility in utility_types_sorted:
            group_data = numeric_df[numeric_df['utility'] == utility][col].dropna()
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
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types_sorted)}

    for col in selected_columns:
        # Create new figure for each plot with increased size
        plt.figure(figsize=(14, 8))  # Increased figure size for better clarity

        means = [summary_data[col][utility]['mean'] for utility in utility_types_sorted]
        cis = [summary_data[col][utility]['ci'] for utility in utility_types_sorted]
        x_pos = np.arange(len(utility_types_sorted))

        # Handle cases where mean or CI might be NaN
        means_plot = [m if not math.isnan(m) else 0 for m in means]
        cis_plot = [c if not math.isnan(c) else 0 for c in cis]

        # Create bar plot with error bars
        plt.bar(x_pos, means_plot, yerr=cis_plot, align='center', alpha=0.7, ecolor='black',
                capsize=10, color=[utility_colors[utility] for utility in utility_types_sorted])

        # Set labels using the conditional logic
        ylabel = set_y_label(col)
        plt.ylabel(ylabel, fontsize=text_size)

        # Set x-axis labels with specified text size
        plt.xticks(x_pos, utility_types_sorted, rotation=45, ha='right', fontsize=text_size)

        # Set y-axis tick labels with specified text size
        plt.yticks(fontsize=text_size)

        # Adjust layout to avoid overlap issues with larger text
        plt.tight_layout()

        # Save the figure as PDF with high resolution
        if t_gpu_min is not None and t_gpu_max is not None:
            t_gpu_range_label = f"{t_gpu_min}_{t_gpu_max}"
        else:
            t_gpu_range_label = "all"

        output_filename = f"{csv_file}_{col}_{t_gpu_range_label}_{label}"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path+'.png')
        # tikzplotlib.save(output_path+'.tex')
        plt.close()
        print(f"Confidence interval plot for '{col}' saved to '{output_path}'.")

def plot_cdf_by_utility(csv_file, label, t_gpu_min=None, t_gpu_max=None, 
                        output_dir='plots/cdf_plots', text_size=25, remove_outliers=True):
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

    # Remove outliers if flag is set
    if remove_outliers:
        df = remove_outliers_iqr(df, selected_columns)

    numeric_df = df[selected_columns].copy()

    # Convert tot_unassigned to percentage
    numeric_df['tot_unassigned'] = (numeric_df['tot_unassigned'] / 70) * 100

    # Combine the selected numerical columns with the 'utility' column
    numeric_df['utility'] = df['utility']

    # Get unique utility types after renaming
    utility_types = numeric_df['utility'].unique()
    print(f"Utility Types after renaming: {utility_types}")
    utility_types_sorted = sorted(utility_types)
    num_utilities = len(utility_types_sorted)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for different utilities
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    utility_colors = {utility: colors[i % len(colors)] for i, utility in enumerate(utility_types_sorted)}

    for col in selected_columns:
        # Create new figure for each plot with increased size
        plt.figure(figsize=(14, 7))  # Increased figure size for better clarity

        for utility in utility_types_sorted:
            group_data = numeric_df[numeric_df['utility'] == utility][col].dropna()
            if group_data.empty:
                print(f"Warning: No valid data for utility '{utility}' in column '{col}'. Skipping.")
                continue
            sorted_data = np.sort(group_data)
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            plt.plot(sorted_data, cdf, label=utility, color=utility_colors[utility], linewidth=2)

        # Set labels and title
        xlabel = set_y_label(col)
        ylabel = 'CDF'
        plt.xlabel(xlabel, fontsize=text_size)
        plt.ylabel(ylabel, fontsize=text_size)

        # Set legend with specified text size
        plt.legend(title='Utility Type', fontsize=text_size, title_fontsize=text_size)

        # Set tick parameters
        plt.xticks(fontsize=text_size)
        plt.yticks(fontsize=text_size)

        # Adjust layout to avoid overlap issues with larger text
        plt.tight_layout()

        # Save the figure as PDF with high resolution
        if t_gpu_min is not None and t_gpu_max is not None:
            t_gpu_range_label = f"{t_gpu_min}_{t_gpu_max}"
        else:
            t_gpu_range_label = "all"

        output_filename = f"{csv_file}_{col}_CDF_{t_gpu_range_label}_{label}"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path+'.png')
        # tikzplotlib.save(output_path+'.tex')
        plt.close()
        print(f"CDF plot for '{col}' saved to '{output_path}'.")

if __name__ == "__main__":
    # Flag to determine whether to remove outliers
    REMOVE_OUTLIERS = False

    # Variable to set the text size of all plots
    text_size = 35  # Adjust as needed for better readability

    # Flag to determine whether to use manual intervals
    USE_MANUAL_INTERVALS = True  # Set to False to use percentile-based intervals

    # Define manual t_gpu intervals if USE_MANUAL_INTERVALS is True
    MANUAL_T_GPU_INTERVALS = [
        (0,700),
        (700, 2000)
    ]

    # Variable to set the number of intervals for percentile-based computation
    num_intervals = 2

    # List of CSV file paths without the '.csv' extension
    csv_files = [
        '70J_50N_NFD_HN_NDJ_SPS_NBW_test_results',
        '70J_50N_NFD_HN_NDJ_SPS_BW_test_results',
        '70J_50N_NFD_HN_NDJ_MPS_BW_test_results'
    ]

    # Define labels for each CSV file for identification in plots
    labels = [
        'SPS_NBW',
        'SPS_BW',
        'MPS_BW'
    ]

    # Ensure the number of labels matches the number of CSV files
    assert len(csv_files) == len(labels), "Number of labels must match number of CSV files."

    # Compute t_gpu intervals for your data
    t_gpu_intervals = []

    for csv_file in csv_files:
        if USE_MANUAL_INTERVALS:
            intervals = MANUAL_T_GPU_INTERVALS
            print(f"Using manual t_gpu intervals for '{csv_file}': {intervals}")
        else:
            try:
                df_main = pd.read_csv(csv_file + '.csv')
            except FileNotFoundError:
                print(f"Error: File '{csv_file}.csv' not found.")
                t_gpu_intervals.append([])  # Append empty list for consistency
                continue
            except pd.errors.EmptyDataError:
                print("Error: CSV file is empty.")
                t_gpu_intervals.append([])
                continue
            except pd.errors.ParserError:
                print("Error: CSV file is malformed.")
                t_gpu_intervals.append([])
                continue
            csv_file = csv_file[csv_file['execution']<40]
            # Check if 't_gpu' column exists
            if 't_gpu' not in df_main.columns:
                print(f"Error: 't_gpu' column not found in '{csv_file}.csv'.")
                t_gpu_intervals.append([])
                continue

            # Drop NaN in 't_gpu'
            t_gpu_data = df_main['t_gpu'].dropna()

            if t_gpu_data.empty:
                print(f"Error: 't_gpu' column in '{csv_file}.csv' contains only NaN values.")
                t_gpu_intervals.append([])
                continue

            # Compute percentiles based on num_intervals
            percentiles = np.linspace(0, 100, num_intervals + 1)
            percentile_values = t_gpu_data.quantile(percentiles / 100).values

            print(f"Computed t_gpu Percentiles for '{csv_file}':")
            for p, val in zip(percentiles, percentile_values):
                print(f"{p}th Percentile: {val}")
            print()

            # Define t_gpu_ranges based on percentiles
            intervals = []
            for i in range(num_intervals):
                lower = percentile_values[i]
                upper = percentile_values[i + 1]
                intervals.append((lower, upper))
        # Append the intervals to t_gpu_intervals
        t_gpu_intervals.append(intervals)

    # Loop through each CSV file and its corresponding label
    for csv_file, label, intervals in zip(csv_files, labels, t_gpu_intervals):
        if not intervals:
            print(f"Skipping plotting for '{csv_file}' due to previous errors.")
            continue

        for idx, (t_gpu_min, t_gpu_max) in enumerate(intervals, start=1):
            print(f"Processing '{csv_file}' for t_gpu interval {idx}: {t_gpu_min} to {t_gpu_max}")

            # Plot Confidence Intervals
            plot_confidence_intervals_by_utility(
                csv_file=csv_file,
                label=label,
                confidence=0.95,
                t_gpu_min=t_gpu_min,
                t_gpu_max=t_gpu_max,
                output_dir='plots/confidence_intervals',
                text_size=text_size,
                remove_outliers=REMOVE_OUTLIERS
            )

            # Plot CDFs
            plot_cdf_by_utility(
                csv_file=csv_file,
                label=label,
                t_gpu_min=t_gpu_min,
                t_gpu_max=t_gpu_max,
                output_dir='plots/cdf_plots',
                text_size=text_size,
                remove_outliers=REMOVE_OUTLIERS
            )

    print("All plots generated successfully.")
