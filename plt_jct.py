import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import numpy as np
import math
from itertools import chain, combinations

# Define the utility type mapping
UTILITY_MAPPING = {
    'Utility.UTIL': 'FRAG',
    'Utility.SGF': 'SGF',
    'Utility.LGF': 'LGF',
    'Utility.SEQ': 'SEQ',
    'Utility.LIKELIHOOD': 'LIKELIHOOD',
    'Utility.DRF': 'DRF',
    'Utility.TETRIS': 'TETRIS'
}

# Define utility types for consistency
UTILITY_TYPES = [
    'FRAG', 'SGF', 'LGF', 'SEQ', 'LIKELIHOOD', 'DRF', 'TETRIS'
]

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

def plot_cdf_by_quartile(plot_df, output_dir='plots/cdf_quartiles'):
    """
    Plot the Cumulative Distribution Function (CDF) of Job Completion Time (JCT) grouped by quartiles.

    Parameters:
        plot_df (pd.DataFrame): DataFrame containing 'Quartile' and 'Current Duration' columns.
        output_dir (str): Directory to save the plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set the plot style for better aesthetics
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Plot the CDF for each quartile
    sns.ecdfplot(data=plot_df, x='Current Duration', hue='Quartile', palette='Set1')

    # Customize the plot
    plt.xlabel('Job Completion Time (Duration)', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.title('CDF of Job Completion Time (JCT) by Quartile of Dataset Size', fontsize=16)
    plt.legend(title='Dataset Size Quartile', fontsize=12, title_fontsize=12)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "jct_cdf_by_quartile.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"CDF plot saved to '{output_path}'.")

def aggregate_data_by_quartile(data_directory, utility_types, experiment_numbers):
    """
    Aggregate current_duration data from multiple CSV files, sum num_gpu per file,
    assign quartiles based on the summed num_gpu, and categorize data accordingly.

    Parameters:
        data_directory (str): Directory containing the CSV files.
        utility_types (list): List of utility types.
        experiment_numbers (range): Range of experiment numbers.

    Returns:
        pd.DataFrame: DataFrame containing 'Quartile' and 'Current Duration' columns.
    """
    # Initialize a list to hold data for each file
    data_list = []

    # Iterate through each utility type and experiment number
    print("Collecting data from CSV files...")
    for utility in tqdm(utility_types, desc='Utility Types'):
        for exp_num in tqdm(experiment_numbers, desc=f'Experiment {utility}', leave=False):
            # Construct the filename based on the naming convention
            filename = f'150J_100N_NFD_HN_NDJ_NBW_{exp_num}_{utility}_FIFO_jobs_report.csv'
            file_path = os.path.join(data_directory, filename)

            # Check if the file exists
            if os.path.isfile(file_path):
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Ensure required columns exist
                    if 'current_duration' in df.columns and 'num_gpu' in df.columns:
                        # Convert 'current_duration' and 'num_gpu' to numeric, coercing errors to NaN
                        df['current_duration'] = pd.to_numeric(df['current_duration'], errors='coerce')
                        df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')

                        # Sum 'num_gpu' for the dataset size
                        num_gpu_sum = df['num_gpu'].sum()

                        # Extract 'current_duration' and drop NaNs
                        durations = df['current_duration'].dropna().tolist()

                        # Append to data_list
                        data_list.append({
                            'Utility': utility,
                            'Experiment': exp_num,
                            'Num_GPU_Sum': num_gpu_sum,
                            'Durations': durations
                        })
                    else:
                        print(f"Warning: Required columns not found in {filename}. Skipping.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}. Skipping.")
            else:
                print(f"Warning: File {filename} does not exist. Skipping.")

    # Create a DataFrame from data_list
    data_df = pd.DataFrame(data_list)

    print(f"Total files processed: {len(data_df)}")

    # Check if there are any files
    if data_df.empty:
        print("No data available to process.")
        return pd.DataFrame(columns=['Quartile', 'Current Duration'])

    # Determine quartiles based on 'Num_GPU_Sum'
    quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    try:
        data_df['Quartile'] = pd.qcut(data_df['Num_GPU_Sum'], q=4, labels=quartile_labels)
    except ValueError as e:
        print(f"Error in quartile assignment: {e}")
        # Alternative handling, e.g., using pd.cut with manual binning
        data_df['Quartile'] = pd.cut(data_df['Num_GPU_Sum'], bins=4, labels=quartile_labels)

    # Check distribution
    print("Quartile distribution:")
    print(data_df['Quartile'].value_counts())

    # Initialize a dictionary to hold durations per quartile
    quartile_data = {quartile: [] for quartile in quartile_labels}

    # Populate quartile_data
    for idx, row in data_df.iterrows():
        quartile = row['Quartile']
        quartile_data[quartile].extend(row['Durations'])

    # Prepare data for plotting
    plot_data = []
    for quartile, durations in quartile_data.items():
        for duration in durations:
            plot_data.append({'Quartile': quartile, 'Current Duration': duration})

    # Create a DataFrame
    plot_df = pd.DataFrame(plot_data)

    print("Data aggregation complete.")
    print(plot_df.head())

    return plot_df

def main():
    # Define utility types and experiment numbers
    utility_types = UTILITY_TYPES
    experiment_numbers = range(1, 51)  # 1 to 50 inclusive

    # Define the directory containing the CSV files
    # Replace 'path_to_your_csv_files' with the actual path
    data_directory = '.'  # Example: 'data/csv_reports'

    # Aggregate data by quartile
    plot_df = aggregate_data_by_quartile(
        data_directory=data_directory,
        utility_types=utility_types,
        experiment_numbers=experiment_numbers
    )

    # Check if there's data to plot
    if not plot_df.empty:
        # Plot the CDF by quartile
        plot_cdf_by_quartile(plot_df, output_dir='plots/cdf_quartiles')
    else:
        print("No data available for plotting.")

if __name__ == "__main__":
    main()
