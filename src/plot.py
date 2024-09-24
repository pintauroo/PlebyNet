import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plot_folder(dirname):
    # check if the plot directory exists, if not create it
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def plot_node_resource_usage_box(filename, res_type, n_nodes, dir_name):
    """
    Plots the resource usage of nodes in the form of a boxplot and saves the plot to a file.

    Args:
        filename (str): The name of the file containing the data to plot.
        res_type (str): The type of resource to plot (e.g. "cpu", "gpu").
        n_nodes (int): The number of nodes to plot.
        dir_name (str): The name of the directory to save the plot file in.
    """
    # plot node resource usage using data from filename
    df = pd.read_csv(filename + ".csv")
    
    # select only the columns matching the pattern node_*_updated_gpu
    df2 = df.filter(regex=("node.*"+res_type))
    
    d = {}
    for i in range(n_nodes):
        gpu_type = df['node_'+str(i)+'_gpu_type'].iloc[1]
        if gpu_type not in d:
            d[str(gpu_type)] = []
        d[str(gpu_type)] += list(df2["node_" + str(i) + "_used_" + res_type] / df2["node_" + str(i) + "_initial_" + res_type])
    
    # use matplotlib to plot the data and save the plot to a file
    plt.boxplot(d.values())
    plt.xticks(range(1, len(d.keys()) + 1), d.keys())

    
    plt.ylabel(f"{res_type} usage")
    plt.xlabel("GPU type")
    plt.savefig(os.path.join(dir_name, 'node_' + res_type + '_resource_usage_box.png'))
    # ticks = [i+1 for i in range(len(d.keys()))]
    # plt.xticks(ticks, d.keys())
    
    # clear plot
    plt.clf()
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_node_resource_usage(filename, res_type, n_nodes, dir_name):
    """
    Plots the resource usage of nodes as a bar plot and saves the plot to a file.

    Args:
        filename (str): The name of the file containing the data to plot.
        res_type (str): The type of resource to plot (e.g. "cpu", "gpu").
        n_nodes (int): The number of nodes to plot.
        dir_name (str): The name of the directory to save the plot file in.
    """
    # Ensure the directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Load the data from the CSV file
    df = pd.read_csv(filename + ".csv")
    
    # Select only the columns matching the pattern node_*_used_res_type
    df2 = df.filter(regex=(f"node.*used_{res_type}"))
    
    # Prepare a dictionary to store normalized usage data for each node
    d = {}
    for i in range(n_nodes):
        try:
            gpu_type = df[f'node_{i}_gpu_type'].iloc[0]
            # Calculate the average or sum of resource usage over time for the bar plot
            d[f"node_{i}_{gpu_type}"] = (df[f"node_{i}_used_{res_type}"] / df[f"node_{i}_initial_{res_type}"]).mean()  # Use .mean() to get the average usage
        except KeyError as e:
            print(f"Warning: Column for node {i} and resource {res_type} not found. Skipping this node.")
            continue
    
    # Create a DataFrame from the normalized usage data for bar plotting
    df_2 = pd.DataFrame(list(d.items()), columns=['Node', f'{res_type} Usage'])

    # Generate a bar plot
    df_2.set_index('Node').plot(kind='bar', legend=None)
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_node_resource_usage(filename, res_type, n_nodes, dir_name):
    """
    Plots the resource usage of nodes as a bar plot and saves the plot to a file.

    Args:
        filename (str): The name of the file containing the data to plot.
        res_type (str): The type of resource to plot (e.g. "cpu", "gpu").
        n_nodes (int): The number of nodes to plot.
        dir_name (str): The name of the directory to save the plot file in.
    """
    # Ensure the directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Load the data from the CSV file
    df = pd.read_csv(filename + ".csv")
    
    # Select only the columns matching the pattern node_*_used_res_type
    df2 = df.filter(regex=(f"node.*used_{res_type}"))
    
    # Prepare a dictionary to store normalized usage data for each node
    d = {}
    for i in range(n_nodes):
        try:
            gpu_type = df[f'node_{i}_gpu_type'].iloc[0]
            # Calculate the average or sum of resource usage over time for the bar plot
            d[f"node_{i}_{gpu_type}"] = (df[f"node_{i}_used_{res_type}"] / df[f"node_{i}_initial_{res_type}"]).mean()  # Use .mean() to get the average usage
        except KeyError as e:
            print(f"Warning: Column for node {i} and resource {res_type} not found. Skipping this node.")
            continue
    
    # Create a DataFrame from the normalized usage data for bar plotting
    df_2 = pd.DataFrame(list(d.items()), columns=['Node', f'{res_type} Usage'])

    # Generate a bar plot
    df_2.set_index('Node').plot(kind='bar', legend=None)

    plt.ylabel(f"Average {res_type} usage")
    plt.xlabel("Nodes")
    plt.title(f"Average {res_type} usage across {n_nodes} nodes")

    # Save the plot to a file
    plot_filename = os.path.join(dir_name, f'node_{res_type}_resource_usage_barplot.png')
    plt.savefig(plot_filename)
    
    # Clear plot to free memory
    plt.clf()
    plt.close()
    
    print(f"Bar plot saved to {plot_filename}")


    plt.ylabel(f"Average {res_type} usage")
    plt.xlabel("Nodes")
    plt.title(f"Average {res_type} usage across {n_nodes} nodes")

    # Save the plot to a file
    plot_filename = os.path.join(dir_name, f'node_{res_type}_resource_usage_barplot.png')
    plt.savefig(plot_filename)
    
    # Clear plot to free memory
    plt.clf()
    plt.close()
    
    print(f"Bar plot saved to {plot_filename}")


    
def plot_job_execution_delay(filename, dir_name):
    """
    Plots a histogram of job execution delays and saves the plot to a file.

    Args:
        filename (str): The name of the CSV file containing job data.
        dir_name (str): The name of the directory where the plot will be saved.
    """
    try:
        df = pd.read_csv(filename + "_jobs_report.csv")
    except:
        return
        
    res = df["exec_time"] - df["submit_time"]
        
    # plot histogram using the res variable
    res.astype(int).hist()
    
    # save the plot to a file
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job execution delay (s)")
    plt.savefig(os.path.join(dir_name, 'job_execution_delay.png'))
    
    # clear plot
    plt.clf()
    plt.close()

    
def plot_job_deadline(filename, dir_name):
    """
    Plots a histogram of job deadline exceeded times based on the given CSV file.

    Args:
        filename (str): The name of the CSV file (without the .csv extension).
        dir_name (str): The name of the directory where the plot will be saved.

    Returns:
        None
    """
    try:
        df = pd.read_csv(filename + "_jobs_report.csv")
    except:
        return
        
    res = df["exec_time"] + df["duration"] - df["deadline"]
        
    # plot histogram using the res variable
    res.astype(int).hist()
    
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job deadline exceeded (s)")
    
    # save the plot to a file
    plt.savefig(os.path.join(dir_name, 'job_deadline_exceeded.png'))
    
    # clear plot
    plt.clf()
    plt.close()
    
def plot_power_consumption(filename, res_type, n_nodes, dir_name):
    # plot node resource usage using data from filename
    df = pd.read_csv(filename + ".csv")
    
    # select only the columns matching the pattern node_*_updated_gpu
    df2 = df.filter(regex=("node.*"+res_type))
    
    d = {}
    for i in range(n_nodes):
        gpu_type = df['node_'+str(i)+'_gpu_type'].iloc[0]
        d["node_" + str(i) + "_" + str(gpu_type)] = df2["node_" + str(i) + "_" + res_type + "_consumption"]
    
    df_2 = pd.DataFrame(d)
    
    # use matplotlib to plot the data and save the plot to a file
    df_2.plot(legend=None)
    
    plt.ylabel(f"{res_type} consumption")
    plt.xlabel("time")
    plt.savefig(os.path.join(dir_name, 'node_' + res_type + '_consumption.png'))
    
    # clear plot
    plt.clf()
    plt.close()
    
def plot_job_messages_exchanged(job_count, dir_name):
    """
    Generate a boxplot of the number of messages exchanged by each job and save the plot to a file.

    Args:
        job_count (dict): A dictionary containing the number of messages exchanged by each job.
        dir_name (str): The directory where the plot will be saved.

    Returns:
        None
    """
    data = list(job_count.values())
    
    _ = plt.figure()
 
    # Creating plot
    plt.boxplot(data)
    
    plt.savefig(os.path.join(dir_name, 'number_messages_job.png'))
    
    # clear plot
    plt.clf()
    plt.close()
    
def plot_job_processing_times(processing_times, post_processing_time, dir_name):
    _ = plt.figure()
 
    # Creating plot
    plt.plot(processing_times, label="Job allocation")
    plt.plot(post_processing_time, label="Job post processing")
    plt.legend()
    plt.ylabel("Processing time (s)")
    
    plt.savefig(os.path.join(dir_name, 'job_processing_times.png'))
    
    # clear plot
    plt.clf()
    plt.close()
    

    
def plot_all(n_edges, filename, job_count, dir_name, processing_times=[], post_process_time=[]):
    """
    Plots all the relevant graphs for the given parameters.

    Args:
        n_edges (int): Number of edges in the graph.
        filename (str): Name of the file containing the data.
        job_count (dict): Jobs in the system.
        dir_name (str): Name of the directory where the plots will be saved.
    """
    generate_plot_folder(dir_name)
    
    plot_node_resource_usage(filename, "gpu", n_edges, dir_name)
    plot_node_resource_usage(filename, "cpu", n_edges, dir_name)
    plot_node_resource_usage(filename, "bw", n_edges, dir_name)
    
    plot_node_resource_usage_box(filename, "gpu", n_edges, dir_name)
    plot_node_resource_usage_box(filename, "cpu", n_edges, dir_name)
    plot_node_resource_usage_box(filename, "bw", n_edges, dir_name)
    
    plot_power_consumption(filename, "cpu", n_edges, dir_name)
    plot_power_consumption(filename, "gpu", n_edges, dir_name)
    
    plot_job_execution_delay(filename, dir_name)
    plot_job_deadline(filename, dir_name)
    
    plot_job_messages_exchanged(job_count, dir_name)
    plot_job_processing_times(processing_times, post_process_time, dir_name)
    
def generate_plots():
    dir_name = "plot"
    generate_plot_folder(dir_name)
    filename = '/home/andrea/PlebyNet/0_LGF_FIFO_1_nosplit_norebid'

    plot_node_resource_usage(filename, "gpu", 100, dir_name)
    plot_node_resource_usage(filename, "cpu", 100, dir_name)

    # plot_node_resource_usage_box("GPU", "gpu", 100, dir_name)
    # plot_node_resource_usage_box("GPU", "cpu", 100, dir_name)

    # plot_job_execution_delay("jobs_report", dir_name)
    # plot_job_deadline("jobs_report", dir_name)

if __name__ == "__main__":
    generate_plots()
