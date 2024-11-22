import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
import numpy as np

# Define a variable for text size
TEXT_SIZE = 24  # You can adjust this value as needed

# Replace 'data.csv' with the path to your CSV file
csv_file = '/home/fesposito/Andrea/tst/PlebyNet/traces/cleaned_dfws.csv'

# Read the CSV file
try:
    df = pd.read_csv(csv_file)
except pd.errors.EmptyDataError:
    print("The CSV file is empty.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the CSV file. Please check the file format.")
    exit()
except FileNotFoundError:
    print(f"The file {csv_file} does not exist.")
    exit()

# Display the first few rows to verify
print(df.head())

# Check if necessary columns exist
required_columns = ['net_write', 'inst_num', 'plan_cpu', 'plan_gpu']
for col in required_columns:
    if col not in df.columns:
        print(f"The '{col}' column is not found in the CSV file.")
        exit()

# Drop rows with missing values in necessary columns
df = df.dropna(subset=required_columns)

# Ensure columns are numeric
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where conversion to numeric failed
df = df.dropna(subset=required_columns)

# ============================
# Matplotlib Plotting Section
# ============================

# Create a function to plot histograms without log x-axis
def plot_histogram(data, column, title, xlabel, filename_tex, filename_png):
    # Create a figure with specified size
    plt.figure(figsize=(9, 3))

    # Plot histogram as percentages
    weights = (1 / len(data)) * 100  # Normalize to percentages
    plt.hist(data[column], bins=20, color='skyblue', edgecolor='black', weights=[weights] * len(data))

    # Set title and labels with the defined text size
    plt.title(title, fontsize=TEXT_SIZE)
    plt.xlabel(xlabel, fontsize=TEXT_SIZE)
    plt.ylabel('Percentage', fontsize=TEXT_SIZE)

    # Customize tick parameters
    plt.xticks(fontsize=TEXT_SIZE - 2)
    plt.yticks(fontsize=TEXT_SIZE - 2)

    # Remove or comment out the log scale line
    # plt.xscale('log')  # Removed for linear scale

    # Add grid
    plt.grid(axis='y', alpha=0.75)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save as TeX using tikzplotlib with specified axis dimensions
    try:
        tikzplotlib.save(
            filename_tex,
            axis_width='15cm',
            axis_height='10cm'
        )
        print(f"TeX file saved to '{filename_tex}'.")
    except Exception as e:
        print(f"Error saving TeX file '{filename_tex}': {e}")

    # Save the figure
    plt.savefig(filename_png)

    # Clear the current figure to free memory
    plt.clf()

# Plot for 'net_write'
plot_histogram(
    df,
    'net_write',
    'Distribution of net_write',
    'net_write',
    'distrib1_percentage.tex',
    'distrib1_percentage.png'
)

# Plot for 'inst_num'
plot_histogram(
    df,
    'inst_num',
    'Distribution of inst_num',
    'inst_num',
    'distrib_inst_num.tex',
    'distrib_inst_num.png'
)

# Plot for 'plan_cpu'
plot_histogram(
    df,
    'plan_cpu',
    'Distribution of plan_cpu',
    'plan_cpu',
    'distrib_plan_cpu.tex',
    'distrib_plan_cpu.png'
)

# Plot for 'plan_gpu'
plot_histogram(
    df,
    'plan_gpu',
    'Distribution of plan_gpu',
    'plan_gpu',
    'distrib_plan_gpu.tex',
    'distrib_plan_gpu.png'
)

# ============================
# Seaborn Plotting Section
# ============================

def plot_seaborn_histogram(data, column, xlabel, filename_tex, filename_png):
    # Set Seaborn style and context with the defined text size
    sns.set(style="whitegrid", context="notebook", font_scale=TEXT_SIZE / 10)

    # Create a figure with specified size
    plt.figure(figsize=(12, 4))

    # Plot histogram with KDE, normalizing to show percentages
    sns.histplot(data[column], bins=20, kde=True, color='teal', stat="percent")

    # Set title if desired (optional)
    # plt.title(f'Distribution of {column}', fontsize=TEXT_SIZE)

    # Set labels
    plt.xlabel(xlabel, fontsize=TEXT_SIZE)
    plt.ylabel('Percentage', fontsize=TEXT_SIZE)

    # Customize tick parameters
    plt.xticks(fontsize=TEXT_SIZE - 2)
    plt.yticks(fontsize=TEXT_SIZE - 2)

    # Remove or comment out the log scale line
    # plt.xscale('log')  # Removed for linear scale

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename_png)
    
    # Save as TeX using tikzplotlib with default axis dimensions
    try:
        tikzplotlib.save(
            filename_tex
        )
        print(f"TeX file saved to '{filename_tex}'.")
    except Exception as e:
        print(f"Error saving TeX file '{filename_tex}': {e}")

    # Clear the current figure to free memory
    plt.clf()

# Plot for 'net_write'
plot_seaborn_histogram(
    df,
    'net_write',
    'BW utilization (Gbps)',
    'distrib_percentage.tex',
    'distrib_percentage.png'
)

# Plot for 'inst_num'
plot_seaborn_histogram(
    df,
    'inst_num',
    'inst_num',
    'distrib_inst_num_seaborn.tex',
    'distrib_inst_num_seaborn.png'
)

# Plot for 'plan_cpu'
plot_seaborn_histogram(
    df,
    'plan_cpu',
    'plan_cpu',
    'distrib_plan_cpu_seaborn.tex',
    'distrib_plan_cpu_seaborn.png'
)

# Plot for 'plan_gpu'
plot_seaborn_histogram(
    df,
    'plan_gpu',
    'plan_gpu',
    'distrib_plan_gpu_seaborn.tex',
    'distrib_plan_gpu_seaborn.png'
)
