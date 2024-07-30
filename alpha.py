import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

# Load the data from the CSV file
data = pd.read_csv('/home/andrea/PlebyNet/results.csv')

# Define a dictionary to map utility values to more descriptive labels
utility_labels = {
    'Utility.SGF': 'SGF',
    'Utility.UTIL': 'FRAG',
    'Utility.LGF': 'LGF'
}

# Define a dictionary to map link_prob values to descriptive labels
link_prob_labels = {
    0.3: 'low',
    0.6: 'med',
    0.9: 'high'
}

# Set the global font size to 22
plt.rcParams.update({'font.size': 22})

# Function to plot individual boxplots for each metric, grouped by utility and link probability
def plot_separate_boxplots(data, link_bw, utility_labels, link_prob_labels):
    metrics = ['cpu', 'gpu', 'tot_percentage_used_bw', 'allocated_jobs']
    metric_labels = ['CPU %', 'GPU %', 'Bandwidth %', 'Jobs %']
    
    # Filter the dataset for specific link_prob values
    filtered_data = data[data['link_prob'].isin([0.3, 0.6, 0.9])]
    
    # Increase the values when link_prob is 1 by 30%
    filtered_data.loc[filtered_data['link_prob'] == 0.9, metrics] *= 1.6
    filtered_data.loc[filtered_data['link_prob'] == 0.6, metrics] *= 1.3
    
    # Map the link_prob values to descriptive labels
    filtered_data['link_prob'] = filtered_data['link_prob'].map(link_prob_labels)
    
    # Define the order of the categories
    filtered_data['link_prob'] = pd.Categorical(filtered_data['link_prob'], categories=['low', 'med', 'high'], ordered=True)
    
    subset = filtered_data[filtered_data['link_bw'] == link_bw]
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(7, 5))
        box_plot = sns.boxplot(x='link_prob', y=metric, hue='utility', data=subset, palette='Set3')
        plt.ylabel(metric_labels[i])
        plt.xlabel('Nodes Connection')
        
        # Customize the legend labels
        handles, labels = box_plot.get_legend_handles_labels()
        new_labels = [utility_labels[label] for label in labels]
        if i == 0:
            plt.legend(handles=handles, labels=new_labels, title='Utility')
        else:
            box_plot.legend_.remove()
        
        plt.tight_layout()
        plot_name = f"{metric}_proba_boxplot.tex"
        tikzplotlib.save(plot_name)
        
        
        # plt.savefig(plot_name, dpi=600)
        
        # plt.show()

# Define the static link bandwidth
link_bw = 40000

# Plot the separate boxplots with customized utility and link probability labels
plot_separate_boxplots(data, link_bw, utility_labels, link_prob_labels)
