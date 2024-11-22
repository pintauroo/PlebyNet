import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import os

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
        ylabel = 'First Failure (%)'
    elif col == 'tot_unassigned':
        ylabel = 'Allocation Failure Rate (%)'
    elif col == 'jct_mean':
        ylabel = 'JCT Mean'
    elif col == 'jct_median':
        ylabel = 'JCT Median'
    else:
        ylabel = col
    return ylabel

def plot_combined_confidence_intervals(csv_files, labels, confidence=0.95, t_gpu_min=None, t_gpu_max=None, 
                                       output_dir='plots/combined_confidence_intervals', text_size=12, remove_outliers=True):
    # Ensure the number of labels matches the number of CSV files
    assert len(csv_files) == len(labels), "Number of labels must match number of CSV files."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")
    
    # Initialize a list to hold data from all CSV files
    all_data = []
    
    # Set global font size using rcParams
    plt.rcParams.update({'font.size': text_size})
    
    # Loop through each CSV file and its corresponding label
    for csv_file, label in zip(csv_files, labels):
        print(f"\nProcessing file: {csv_file}.csv with label: {label}")
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file + '.csv')
            print(f"Successfully read '{csv_file}.csv'.")
        except FileNotFoundError:
            print(f"Error: File '{csv_file}.csv' not found. Skipping this file.")
            continue
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file '{csv_file}.csv' is empty. Skipping this file.")
            continue
        except pd.errors.ParserError:
            print(f"Error: CSV file '{csv_file}.csv' is malformed. Skipping this file.")
            continue

        # Apply t_gpu filters if specified
        if t_gpu_min is not None:
            df = df[df['t_gpu'] >= t_gpu_min]
            print(f"Applied t_gpu_min filter: >= {t_gpu_min}")
        if t_gpu_max is not None:
            df = df[df['t_gpu'] < t_gpu_max]
            print(f"Applied t_gpu_max filter: < {t_gpu_max}")
        print(f"Number of rows after t_gpu filtering: {len(df)}")

        # Ensure 'utility' column exists
        if 'utility' not in df.columns:
            print(f"Error: 'utility' column not found in the CSV file '{csv_file}'. Skipping this file.")
            continue

        # Drop rows where 'utility' is NaN
        initial_row_count = len(df)
        df = df.dropna(subset=['utility'])
        final_row_count = len(df)
        dropped_rows = initial_row_count - final_row_count
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to NaN in 'utility' column in '{csv_file}.csv'.")

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
            print(f"Warning: Found unmapped utility types in '{csv_file}.csv': {unique_unmapped}")
            df = df.dropna(subset=['utility'])
            print(f"Dropped rows with unmapped utility types in '{csv_file}.csv'.")

        # Select the specified numerical columns
        selected_columns = [
            'first_unassigned_gpu', 'first_unassigned_cpu', 'first_unassigned', 
            'jct_mean', 'jct_median', 'tot_unassigned', 'discarded_jobs'
        ]

        # Verify that the selected columns exist in the dataframe
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: The following required columns are missing in the CSV file '{csv_file}.csv': {missing_columns}. Skipping this file.")
            continue

        # Remove outliers if flag is set
        if remove_outliers:
            df = remove_outliers_iqr(df, selected_columns)

        numeric_df = df[selected_columns].copy()

        # Convert tot_unassigned to percentage
        if 'tot_unassigned' in numeric_df.columns:
            numeric_df['tot_unassigned'] = (numeric_df['tot_unassigned'] / 70) * 100
            print(f"Converted 'tot_unassigned' to percentage.")

        # Combine the selected numerical columns with the 'utility' column
        numeric_df['utility'] = df['utility']
        numeric_df['experiment'] = label  # Add a column to identify the experiment

        # Append to all_data list
        all_data.append(numeric_df)
        print(f"Data from '{csv_file}.csv' with label '{label}' appended for plotting.")

    # Concatenate all dataframes
    if not all_data:
        print("No data available to plot. Exiting plotting function.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined DataFrame created with {len(combined_df)} rows.")

    # Get unique utility types and experiments
    utility_types = sorted(combined_df['utility'].unique())
    experiments = labels
    num_utilities = len(utility_types)
    num_experiments = len(experiments)
    print(f"Utility Types after renaming: {utility_types}")
    print(f"Experiments: {experiments}")

    # Compute mean and confidence intervals for each numerical column grouped by utility and experiment
    summary_data = {}
    for col in selected_columns:
        summary_data[col] = {}
        for utility in utility_types:
            summary_data[col][utility] = {}
            for experiment in experiments:
                group_data = combined_df[
                    (combined_df['utility'] == utility) & (combined_df['experiment'] == experiment)
                ][col].dropna()
                if group_data.empty:
                    print(f"Warning: No valid data for utility '{utility}' and experiment '{experiment}' in column '{col}'.")
                    mean = np.nan
                    ci = np.nan
                else:
                    mean = group_data.mean()
                    ci = compute_confidence_interval(group_data, confidence=confidence)
                summary_data[col][utility][experiment] = {'mean': mean, 'ci': ci}

    # Define colors for different experiments
    color_map = plt.get_cmap('tab10')
    colors = color_map.colors
    experiment_colors = {experiment: colors[i % len(colors)] for i, experiment in enumerate(experiments)}
    print(f"Assigned colors to experiments: {experiment_colors}")

    for col in selected_columns:
        print(f"\nCreating plot for column: '{col}'")
        # Create new figure for each plot with increased height to accommodate legend
        fig_width, fig_height = 10, 6  # Adjusted width and height
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        bar_width = 0.2
        x = np.arange(len(utility_types))

        for idx, experiment in enumerate(experiments):
            means = [summary_data[col][utility][experiment]['mean'] for utility in utility_types]
            cis = [summary_data[col][utility][experiment]['ci'] for utility in utility_types]
            
            # Handle cases where mean or CI might be NaN
            means_plot = [m if not math.isnan(m) else 0 for m in means]
            cis_plot = [c if not math.isnan(c) else 0 for c in cis]

            # Plot the bars without error bars
            ax.bar(
                x + idx * bar_width, 
                means_plot, 
                width=bar_width, 
                align='center', 
                alpha=0.7, 
                label=experiment, 
                color=experiment_colors[experiment]
            )

            # Add error bars separately
            ax.errorbar(
                x + idx * bar_width, 
                means_plot, 
                yerr=cis_plot, 
                fmt='none', 
                ecolor='black', 
                capsize=5
            )

        # Set labels using the conditional logic
        ylabel = set_y_label(col)
        ax.set_ylabel(ylabel, fontsize=text_size)

        # Set x-axis labels with specified text size
        ax.set_xticks(x + bar_width * (num_experiments - 1) / 2)
        ax.set_xticklabels(utility_types, rotation=45, ha='right', fontsize=text_size)

        # Set y-axis tick labels with specified text size
        ax.tick_params(axis='y', labelsize=text_size)

        # Set legend with specified text size, position it at the top inside the axis, and arrange in a row
        ax.legend(fontsize=text_size, loc='upper center', ncol=num_experiments, bbox_to_anchor=(0.5, 1.3))

        # Adjust layout to make space for the legend
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Increase top margin to accommodate legend inside the plot

        # Save the figure as SVG
        try:
            if t_gpu_min is not None and t_gpu_max is not None:
                t_gpu_range_label = f"{t_gpu_min}_{t_gpu_max}"
            else:
                t_gpu_range_label = "all"

            output_filename = f"combined_{col}_{t_gpu_range_label}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as SVG
            plt.savefig(output_path + '.svg', format='svg')
            plt.savefig(output_path + '.png',)
            print(f"SVG image saved to '{output_path}.svg'.")

        except Exception as e:
            print(f"Error saving SVG image '{output_path}.svg': {e}")

        plt.close(fig)
        print(f"Combined confidence interval plot for '{col}' saved to '{output_path}.svg'.\n")

def plot_cdf_by_utility(csv_file, label, t_gpu_min=None, t_gpu_max=None, 
                        output_dir='plots/cdf_plots', text_size=12, remove_outliers=True):
    # This function remains unchanged from your original code
    pass  # Placeholder if you intend to use it alongside the new combined plots

if __name__ == "__main__":
    # Flag to determine whether to remove outliers
    REMOVE_OUTLIERS = False

    # Variable to set the text size of all plots
    text_size = 25  # Standard LaTeX font sizes are 10pt, 11pt, or 12pt

    # Flag to determine whether to use manual intervals
    USE_MANUAL_INTERVALS = True  # Set to False to use percentile-based intervals

    # Define manual t_gpu intervals if USE_MANUAL_INTERVALS is True
    MANUAL_T_GPU_INTERVALS = [
        (0, 700),
        (700, 2000)
    ]

    # Variable to set the number of intervals for percentile-based computation
    num_intervals = 2

    # List of CSV file paths without the '.csv' extension
    csv_files = [
        '70J_50N_NFD_HN_NDJ_MPS_BW_test_results',  # MPS
        '70J_50N_NFD_HN_NDJ_SPS_BW_test_results',  # SPS
        '70J_50N_NFD_HN_NDJ_SPS_NBW_test_results'  # NBW
    ]

    # Define labels for each CSV file for identification in plots
    labels = [
        'MPS',
        'SPS',
        'NBW'
    ]

    # Ensure the number of labels matches the number of CSV files
    assert len(csv_files) == len(labels), "Number of labels must match number of CSV files."

    # Compute t_gpu intervals for your data
    t_gpu_intervals = []

    if USE_MANUAL_INTERVALS:
        intervals = MANUAL_T_GPU_INTERVALS
        print(f"Using manual t_gpu intervals: {intervals}")
    else:
        # Implement percentile-based interval computation if needed
        intervals = []  # Placeholder for percentile-based intervals

    # Loop through each interval
    for idx, (t_gpu_min, t_gpu_max) in enumerate(intervals, start=1):
        print(f"\nProcessing t_gpu interval {idx}: {t_gpu_min} to {t_gpu_max}")

        # Plot Combined Confidence Intervals
        plot_combined_confidence_intervals(
            csv_files=csv_files,
            labels=labels,
            confidence=0.95,
            t_gpu_min=t_gpu_min,
            t_gpu_max=t_gpu_max,
            output_dir='plots/combined_confidence_intervals',
            text_size=text_size,
            remove_outliers=REMOVE_OUTLIERS
        )

    # If no intervals are defined, plot for all data
    if not intervals:
        print("\nNo t_gpu intervals defined. Plotting for all data.")
        plot_combined_confidence_intervals(
            csv_files=csv_files,
            labels=labels,
            confidence=0.95,
            t_gpu_min=None,
            t_gpu_max=None,
            output_dir='plots/combined_confidence_intervals',
            text_size=text_size,
            remove_outliers=REMOVE_OUTLIERS
        )

    print("\nAll combined plots generated successfully.")
