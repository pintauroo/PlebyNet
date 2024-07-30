import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib


# Load the data from the CSV file
data = pd.read_csv('/home/andrea/PlebyNet/results.csv')

# Concatenate the data (if necessary)
# data = pd.concat([data, data1, data2, data3])
data = pd.concat([data])

# Divide the 'link_bw' values by 1000 to convert the range
data['link_bw'] = (data['link_bw'] / 1000).astype(int)

# Define a dictionary to map utility values to more descriptive labels
utility_labels = {
    'Utility.SGF': 'SGF',
    'Utility.UTIL': 'FRAG',
    'Utility.LGF': 'LGF'
}

# Function to plot individual boxplots for each metric, grouped by utility and bandwidth
def plot_separate_boxplots(data, link_prob, utility_labels, font_size=14):
    metrics = ['cpu', 'gpu', 'tot_percentage_used_bw', 'allocated_jobs']
    metric_labels = ['CPU %', 'GPU %', 'Bandwidth %', ' Jobs %']
    
    subset = data[(data['link_prob'] == link_prob) & (data['link_bw'] < 100)]
    
    # Set font sizes for the plots
    plt.rc('font', size=font_size)         # controls default text size
    plt.rc('axes', titlesize=font_size)  # fontsize of the title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size) # fontsize of the x tick labels
    plt.rc('ytick', labelsize=font_size) # fontsize of the y tick labels
    plt.rc('legend', fontsize=font_size) # fontsize of the legend
    plt.rc('figure', titlesize=font_size) # fontsize of the figure title
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(7, 5))
        box_plot = sns.boxplot(x='link_bw', y=metric, hue='utility', data=subset, palette='Set3')
        plt.ylabel(metric_labels[i])
        plt.xlabel('Link Bandwidth (Gbps)')
        
        # Customize the legend labels
        handles, labels = box_plot.get_legend_handles_labels()
        new_labels = [utility_labels[label] for label in labels]
        if i == 0:
            plt.legend(handles=handles, labels=new_labels, title='Utility')
        else:
            box_plot.legend_.remove()
        
        plt.tight_layout()
        plot_name = f"{metric}_boxplot.tex"
        # plt.savefig(plot_name, dpi=600)
        tikzplotlib.save(plot_name)
        
        plt.show()

# Define the single link probability
link_prob = 0.5

# Set the desired font size
font_size = 22

# Plot the separate boxplots with customized utility labels
plot_separate_boxplots(data, link_prob, utility_labels, font_size)
