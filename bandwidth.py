
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

# Parameters for ResNet-269 training
T_forward_small = 0.3  # Forward propagation time (seconds) for the smallest model size
T_back_small = 0.15  # Backward propagation time (seconds) for the smallest model size
T_forward_medium = 0.5  # Forward propagation time (seconds) for the medium model size
T_back_medium = 0.25  # Backward propagation time (seconds) for the medium model size
T_forward_large = 0.7  # Forward propagation time (seconds) for the largest model size
T_back_large = 0.35  # Backward propagation time (seconds) for the largest model size

w = 4  # Number of workers
p = 2  # Number of parameter servers
T_update = 0.05  # Parameter update time (seconds) - assumed to be constant
delta = 0.001  # Communication overhead coefficient for workers
delta_prime = 0.002  # Communication overhead coefficient for parameter servers

# Model sizes in bytes (e.g., small, medium, large)
model_sizes = [5e8, 1.5e9, 3e9]  # 500 MB, 1.5 GB, 3 GB

# Bandwidth capacities to simulate congestion
bandwidths_gbps = np.array([10, 50, 100])  # From 0.1 Gbps to 10 Gbps

# Convert bandwidths from Gbps to bytes/second
bandwidths_bytes_per_sec = bandwidths_gbps * 1e9 / 8

# Number of workers per parameter server
w_prime = w / p

# Function to calculate different components of training time for a given bandwidth and model size
def calculate_training_time_components(B, S, T_forward, T_back):
    rho = B / w_prime
    gradient_size = S / p  # Each parameter server handles a fraction of the model
    data_transfer_time = 2 * gradient_size * w_prime / (p * B)
    update_time = T_update * w_prime / p
    communication_overhead = delta * w + delta_prime * p
    total_time = T_forward + T_back + data_transfer_time + update_time + communication_overhead
    return {
        'total': total_time,
        'training': T_forward + T_back,
        'data_transfer': data_transfer_time,
        'update': update_time,
        'communication': communication_overhead
    }

# Training times for different model sizes
training_params = {
    model_sizes[0]: (T_forward_small, T_back_small),
    model_sizes[1]: (T_forward_medium, T_back_medium),
    model_sizes[2]: (T_forward_large, T_back_large)
}

# Calculate training times for different bandwidths and model sizes
training_times = {
    model_size: [calculate_training_time_components(B, model_size, *training_params[model_size]) for B in bandwidths_bytes_per_sec]
    for model_size in model_sizes
}

# Extract different components
def extract_components(training_times):
    total_times = [t['total'] for t in training_times]
    training_times_merged = [t['training'] for t in training_times]
    data_transfer_times = [t['data_transfer'] for t in training_times]
    update_times = [t['update'] for t in training_times]
    communication_times = [t['communication'] for t in training_times]
    return total_times, training_times_merged, data_transfer_times, update_times, communication_times

# Define lighter colors and hatches for bars
colors = ['#aec7e8', '#ffbb78', '#98df8a']
hatches = ['/', '\\', '|']
component_colors = ['#c5b0d5', '#f7b6d2', '#c7c7c7']

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 3.5))

# Bar height and positions
bar_height = 0.15
indices = np.arange(len(bandwidths_gbps))
n_models = len(model_sizes)

model_names = ['ResNet 102M', 'Bert 340M', 'GPT 762M']
for i, model_size in enumerate(model_sizes):
    total_times, training_times_merged, data_transfer_times, update_times, communication_times = extract_components(training_times[model_size])

    # Calculate positions for the bars of this model size
    positions = indices + i * bar_height

    # Plot each component as a stacked bar
    ax.barh(positions, training_times_merged, bar_height, label=f'MBatch Training Time ({model_names[i]})', color=colors[i], hatch=hatches[i], edgecolor='black')
    # ax.barh(positions, training_times_merged, bar_height, label=f'Training Time (Model {model_size / 1e9:.1f} GB)', color=colors[i], hatch=hatches[i], edgecolor='black')
    ax.barh(positions, data_transfer_times, bar_height, left=training_times_merged, label='Weight Transfer Time' if i == 0 else "", color=component_colors[0], edgecolor='black')
    ax.barh(positions, update_times, bar_height, left=np.array(training_times_merged) + np.array(data_transfer_times), label='PS Parameter processing Time' if i == 0 else "", color=component_colors[1], edgecolor='black')
    # ax.barh(positions, communication_times, bar_height, left=np.array(training_times_merged) + np.array(data_transfer_times) + np.array(update_times), label='Communication Overhead' if i == 0 else "", color=component_colors[2], edgecolor='black')

# Customize the plot
ax.set_yticks(indices + bar_height * (n_models - 1) / 2)
ax.set_yticklabels([f'{b}' for b in bandwidths_gbps], fontsize=14)
ax.tick_params(axis='y', labelsize=13, labelrotation=90)
ax.set_ylabel('Network Bandwidth (Gbps)', fontsize=14)
ax.set_xlabel('Training Time (seconds)', fontsize=14)
ax.legend(loc='upper right', fontsize=13)
ax.grid(True)

plt.tight_layout()
# plt.savefig('bandwidth.png', dpi=600)
# plt.show()



tikzplotlib.save("bandwidth.tex")
