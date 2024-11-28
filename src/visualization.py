import matplotlib.pyplot as plt
import networkx as nx

def plot_topology(graph, title="Fat-Tree Topology", save_path=None):
    """
    Visualize the Fat-Tree topology using Matplotlib.
    
    Parameters:
        graph (networkx.Graph): The graph representing the Fat-Tree topology.
        title (str): Title of the plot.
        save_path (str): If provided, saves the plot to the specified path.
    """
    # Define positions for each layer (core, aggregation, edge, hosts)
    pos = {}
    core_y = 4
    agg_y = 3
    edge_y = 2
    host_y = 1
    
    # Separate nodes by type for plotting
    core_nodes = [n for n in graph.nodes if 'core' in n]
    agg_nodes = [n for n in graph.nodes if 'agg' in n]
    edge_nodes = [n for n in graph.nodes if 'edge' in n]
    host_nodes = [n for n in graph.nodes if 'host' in n]
    
    # Assign positions
    for i, node in enumerate(core_nodes):
        pos[node] = (i, core_y)
    for i, node in enumerate(agg_nodes):
        pos[node] = (i, agg_y)
    for i, node in enumerate(edge_nodes):
        pos[node] = (i, edge_y)
    for i, node in enumerate(host_nodes):
        pos[node] = (i, host_y)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=[
            'red' if 'core' in n else
            'blue' if 'agg' in n else
            'green' if 'edge' in n else
            'orange' if 'host' in n else 'gray'
            for n in graph.nodes
        ],
        edge_color="gray",
        node_size=500,
        font_size=10
    )
    
    # Title and legend
    plt.title(title)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Core', markersize=10, markerfacecolor='red'),
            plt.Line2D([0], [0], marker='o', color='w', label='Aggregation', markersize=10, markerfacecolor='blue'),
            plt.Line2D([0], [0], marker='o', color='w', label='Edge', markersize=10, markerfacecolor='green'),
            plt.Line2D([0], [0], marker='o', color='w', label='Host', markersize=10, markerfacecolor='orange')
        ],
        loc="upper left"
    )
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()
