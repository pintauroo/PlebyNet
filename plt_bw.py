import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the file pattern
FILE_PATTERN = re.compile(
    r'(\d+)_70J_50N_NFD_HN_NDJ_(SPS|MPS)_(BW|NBW)_(TETRIS|DRF|LIKELIHOOD|SGF|LGF|SEQ)_FIFO_topo\.csv'
)

# Initialize data structure
data = {}

# List all files in the current directory
file_list = os.listdir('.')

for filename in file_list:
    match = FILE_PATTERN.match(filename)
    if match:
        id_str, sps_mps, bw_nbw, utility = match.groups()
        if id_str.startswith('110'):
            # Process this file
            key = sps_mps  # Group by SPS or MPS
            if key not in data:
                data[key] = {}
            if utility not in data[key]:
                data[key][utility] = []
            # Load the CSV
            df = pd.read_csv(filename)
            # Divide relevant columns by 100 to get Gbps
            cols_to_divide = df.columns.difference(['allocation_step', 'job_id'])
            df[cols_to_divide] = df[cols_to_divide] / 100
            # Compute total reserved bandwidth and total bandwidth
            reserved_bw_cols = [col for col in df.columns if '_reserved_bw' in col]
            total_bw_cols = [col for col in df.columns if '_total_bw' in col]
            df['total_reserved_bw'] = df[reserved_bw_cols].sum(axis=1)
            df['total_bw'] = df[total_bw_cols].sum(axis=1)
            df['utilization'] = df['total_reserved_bw'] / df['total_bw']
            # Store the DataFrame
            data[key][utility].append(df)

# Plotting
for sps_mps in data:
    plt.figure(figsize=(10, 6))
    for utility in data[sps_mps]:
        # Concatenate DataFrames for the utility
        df_list = data[sps_mps][utility]
        df = pd.concat(df_list, ignore_index=True)
        # Group by allocation_step and compute mean utilization
        df_grouped = df.groupby('allocation_step')['utilization'].mean().reset_index()
        # Plot
        plt.plot(df_grouped['allocation_step'], df_grouped['utilization'], label=utility)
    plt.xlabel('Allocation Step')
    plt.ylabel('Utilization')
    plt.title(f'Average Utilization over Time - {sps_mps}')
    plt.legend()
    plt.grid(True)
    plt.show()
