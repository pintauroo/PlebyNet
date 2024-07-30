import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


# Generating some dummy data for the plots with different lengths and scales
time_optimus = np.linspace(0, 21000, 90)
time_drf = np.linspace(0, 24000, 100)
time_Plebiscito = np.linspace(0, 19000, 80)

tasks_optimus = np.random.randint(10, 30, size=90)
tasks_drf = np.random.randint(20, 40, size=100)
tasks_Plebiscito = np.random.randint(10, 30, size=80)

GPU_ps_optimus = np.random.randint(40, 90, size=90)
GPU_ps_drf = np.random.randint(60, 100, size=100)
GPU_ps_Plebiscito = np.random.randint(50, 95, size=80)

bandwidth_optimus = np.random.randint(80, 150, size=90)
bandwidth_drf = np.random.randint(80, 180, size=100)
bandwidth_Plebiscito = np.random.randint(80, 100, size=80)

# Common font size for all plots
font_size = 24

# Helper function to format x-axis values
def format_xticks(ax):
    xticks = ax.get_xticks()
    ax.set_xticklabels([f'{int(x/1000)}' for x in xticks], fontsize=font_size)

# Plotting the first graph
plt.figure(figsize=(8, 4))
plt.plot(time_optimus, tasks_optimus, 'b-.', label='Optimus', linewidth=4)
plt.plot(time_drf, tasks_drf, 'r--', label='DRF', linewidth=4)
plt.plot(time_Plebiscito, tasks_Plebiscito, 'g-', label='Plebiscito', linewidth=4)
plt.xlabel('Time (s)', fontsize=font_size)
plt.ylabel('# of Running tasks', fontsize=font_size-2)
plt.legend(fontsize=font_size-2, loc='upper left')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax = plt.gca()
format_xticks(ax)
plt.text(1, -0.15, 'x $10^3$ s', transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='right')
plt.tight_layout()
tikzplotlib.save('time_tasks.tex')

plt.savefig('time_tasks.png', dpi=600)
plt.show()

# Plotting the second graph
plt.figure(figsize=(8, 4))
plt.plot(time_optimus, GPU_ps_optimus, 'b-.', label='Optimus', linewidth=4)
plt.plot(time_drf, GPU_ps_drf, 'r--', label='DRF', linewidth=4)
plt.plot(time_Plebiscito, GPU_ps_Plebiscito, 'g-', label='Plebiscito', linewidth=4)
plt.xlabel('Time (s)', fontsize=font_size)
plt.ylabel('Norm. GPU (%)', fontsize=font_size-2)
plt.legend(fontsize=font_size-2, loc='upper left')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax = plt.gca()
format_xticks(ax)
plt.text(1, -0.15, 'x $10^3$ s', transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='right')
plt.tight_layout()
tikzplotlib.save('time_GPU.tex')

plt.savefig('time_GPU.png', dpi=600)
plt.show()

# Plotting the third graph
plt.figure(figsize=(8, 4))
plt.plot(time_optimus, bandwidth_optimus, 'b-.', label='Optimus', linewidth=4)
plt.plot(time_drf, bandwidth_drf, 'r--', label='DRF', linewidth=4)
plt.plot(time_Plebiscito, bandwidth_Plebiscito, 'g-', label='Plebiscito', linewidth=4)
plt.xlabel('Time (s)', fontsize=font_size)
plt.ylabel('Norm. BW (%)', fontsize=font_size-2)
# plt.legend(fontsize=font_size-2, loc='upper left')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax = plt.gca()
format_xticks(ax)
plt.text(1, -0.15, 'x $10^3$ s', transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='right')
plt.tight_layout()
tikzplotlib.save('time_BW.tex')

plt.savefig('time_BW.png', dpi=600)
plt.show()
