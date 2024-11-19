import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Check if 'net_write' column exists
if 'net_write' not in df.columns:
    print("The 'net_write' column is not found in the CSV file.")
    exit()

# Drop rows with missing 'net_write' values
df = df.dropna(subset=['net_write'])

# Ensure 'net_write' is numeric
df['net_write'] = pd.to_numeric(df['net_write'], errors='coerce')

# Drop rows where 'net_write' could not be converted to numeric
df = df.dropna(subset=['net_write'])

# ============================
# Matplotlib Plotting Section
# ============================

# Create a figure with specified size
plt.figure(figsize=(9, 3))

# Plot histogram as percentages
weights = (1 / len(df)) * 100  # Normalize to percentages
plt.hist(df['net_write'], bins=20, color='skyblue', edgecolor='black', weights=[weights] * len(df))

# Set title and labels with the defined text size
plt.title('Distribution of net_write (Percentage)', fontsize=TEXT_SIZE)
plt.xlabel('net_write', fontsize=TEXT_SIZE)
plt.ylabel('Percentage', fontsize=TEXT_SIZE)

# Customize tick parameters
plt.xticks(fontsize=TEXT_SIZE - 2)
plt.yticks(fontsize=TEXT_SIZE - 2)

# Limit the x-axis to 50
plt.xlim(0, 50)

# Add grid
plt.grid(axis='y', alpha=0.75)

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('distrib1_percentage.png')

# Clear the current figure to free memory
plt.clf()

# ============================
# Seaborn Plotting Section
# ============================

# Set Seaborn style and context with the defined text size
sns.set(style="whitegrid", context="notebook", font_scale=TEXT_SIZE / 10)

# Create a figure with specified size
plt.figure(figsize=(12, 4))

# Plot histogram with KDE, normalizing to show percentages
sns.histplot(df['net_write'], bins=20, kde=True, color='teal', stat="percent")

# Set title and labels
plt.xlabel('BW utilization (Gbps)', fontsize=TEXT_SIZE)
plt.ylabel('Percentage', fontsize=TEXT_SIZE)

# Customize tick parameters
plt.xticks(fontsize=TEXT_SIZE - 2)
plt.yticks(fontsize=TEXT_SIZE - 2)

# Limit the x-axis to 50
plt.xlim(0, 50)

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('distrib_percentage.png')

# Clear the current figure to free memory
plt.clf()
