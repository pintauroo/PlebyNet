import networkx as nx
import matplotlib.pyplot as plt
import random

def create_spine_leaf_topology_with_bandwidth(num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
    G = nx.Graph()
    
    # Add spine switches with bandwidth capacity
    spine_switches = [f"S{i}" for i in range(num_spine)]
    for spine in spine_switches:
        G.add_node(spine, type='spine', bandwidth=spine_bw, reserved_bw=0)
    
    # Add leaf switches with bandwidth capacity
    leaf_switches = [f"L{i}" for i in range(num_leaf)]
    for leaf in leaf_switches:
        G.add_node(leaf, type='leaf', bandwidth=leaf_bw, reserved_bw=0)
    
    # Connect each leaf switch to all spine switches and assign link bandwidth with reserved_bw initialized to 0
    for leaf in leaf_switches:
        for spine in spine_switches:
            G.add_edge(leaf, spine, bandwidth=link_bw_leaf_to_spine, reserved_bw=0)
    
    # Add hosts and connect them to leaf switches, assigning link bandwidth for each host connection
    host_id = 0
    for leaf in leaf_switches:
        for _ in range(num_hosts_per_leaf):
            host = f"H{host_id}"
            G.add_node(host, type='host', reserved_bw=0)  # Initialize reserved_bw for host nodes
            G.add_edge(leaf, host, bandwidth=link_bw_leaf_to_node, reserved_bw=0)  # Link bw from host to leaf
            host_id += 1
    
    return G

def allocate_ps_to_workers(G, ps_node, worker_nodes, required_bw, allow_oversubscription=False):
    """
    Allocate bandwidth between a parameter server (PS) and a list of worker nodes.
    Prevent allocation if bandwidth does not fit and oversubscription is disabled.
    Update bandwidth at the switches only once.
    """
    paths = {}
    allocated_edges = []  # Track which edges have been allocated to roll back if necessary
    allocated_nodes = set()  # Track which nodes have been updated to avoid double updating

    # First, check if enough bandwidth is available for all workers
    for worker in worker_nodes:
        if ps_node != worker:
            try:
                path = nx.shortest_path(G, source=ps_node, target=worker)
                paths[worker] = path  # Store the path for this worker
                print(f"Checking path from PS {ps_node} to worker {worker}: {path}")
                
                # Check bandwidth availability on each link in the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    available_bw = G[u][v]['bandwidth'] - G[u][v]['reserved_bw']
                    if available_bw < required_bw:
                        if not allow_oversubscription:
                            print(f"Not enough bandwidth on link {u}-{v}. Available: {available_bw} Gbps, Required: {required_bw} Gbps")
                            return False  # Abort if any link lacks sufficient bandwidth and oversubscription is not allowed
            
            except nx.NetworkXNoPath:
                print(f"No path found between PS {ps_node} and worker {worker}.")
                return False

    # If bandwidth is available on all paths, proceed to allocate
    for worker, path in paths.items():
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            G[u][v]['reserved_bw'] += required_bw
            allocated_edges.append((u, v))

            # Only update node reserved_bw once per node, and only if it hasn't been updated already
            if u not in allocated_nodes:
                G.nodes[u]['reserved_bw'] += required_bw
                allocated_nodes.add(u)
            if v not in allocated_nodes:
                G.nodes[v]['reserved_bw'] += required_bw
                allocated_nodes.add(v)

            remaining_bw = G[u][v]['bandwidth'] - G[u][v]['reserved_bw']
            if remaining_bw < 0:
                if allow_oversubscription:
                    print(f"Allocated with oversubscription. Remaining bandwidth is negative: {remaining_bw} Gbps on link {u}-{v}")
                else:
                    print(f"Allocation failed: insufficient bandwidth on link {u}-{v} without oversubscription")
                    # Rollback any allocations made before this failure
                    for edge in allocated_edges:
                        u, v = edge
                        G[u][v]['reserved_bw'] -= required_bw
                    for node in allocated_nodes:
                        G.nodes[node]['reserved_bw'] -= required_bw
                    return False

    return True  # Allocation successful


# Example use and visualization of the functions
def plot_node_available_bandwidth(G):
    """
    Plots the available bandwidth at each spine and leaf switch.
    """
    nodes = []
    available_bw = []
    
    for node, data in G.nodes(data=True):
        if data['type'] in ['spine', 'leaf']:  # Only for spine and leaf switches
            total_bw = data['bandwidth']
            reserved_bw = data['reserved_bw']
            avail_bw = total_bw - reserved_bw  # Available bandwidth at this node
            nodes.append(node)
            available_bw.append(avail_bw)
    
    # Plotting available bandwidth at each switch as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(nodes, available_bw, color='green')
    plt.xlabel('Nodes')
    plt.ylabel('Available Bandwidth (Gbps)')
    plt.title('Available Bandwidth at Each Node (Spine and Leaf Switches)')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('plot_node_available_bandwidth.png')

def plot_bandwidth_utilization(G):
    # Collect bandwidth utilization data for each link
    links = []
    utilizations = []
    
    for u, v, data in G.edges(data=True):
        total_bw = data['bandwidth']
        reserved_bw = data['reserved_bw']  # How much bandwidth was reserved
        utilization = (reserved_bw / total_bw) * 100 if total_bw > 0 else 0  # Percentage utilization
        links.append(f"{u}-{v}")
        utilizations.append(utilization)
    
    # Plotting bandwidth utilization as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(links, utilizations, color='skyblue')
    plt.xlabel('Links')
    plt.ylabel('Bandwidth Utilization (%)')
    plt.title('Bandwidth Utilization on Each Link')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('plot_bandwidth_utilization.png')

def plot_degree_distribution(G):
    # Collect degree of each node
    degrees = [G.degree(n) for n in G.nodes()]
    
    # Plotting degree distribution as a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), color='skyblue', alpha=0.7)
    plt.xlabel('Degree (Number of Connections)')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution of Nodes')
    plt.tight_layout()
    plt.savefig('plot_degree_distribution.png')

# Parameters for Medium Deployment
num_spine_switches = 4  # Typically more spines for higher redundancy
num_leaf_switches = 6
num_hosts_per_leaf = 8  # 6 leaves * 8 hosts = 48 servers

# Bandwidth values in Gbps
spine_bandwidth = 100  # Uplinks to spine (100 Gbps)
leaf_bandwidth = 400   # Leaf switches with 4x 100 Gbps uplinks = 400 Gbps
link_bw_leaf_to_node = 100  # 25 Gbps from leaf to node
link_bw_leaf_to_spine = 100  # 100 Gbps uplinks to spine

# Generate the topology with bandwidth
G = create_spine_leaf_topology_with_bandwidth(num_spine_switches, num_leaf_switches, num_hosts_per_leaf, 
                                              spine_bandwidth, leaf_bandwidth, link_bw_leaf_to_node, link_bw_leaf_to_spine)

import random

# Loop to perform bandwidth allocation multiple times with random values
for i in range(10000):  # Change the range as needed for more iterations
    # Generate random PS node and worker nodes
    ps_node = f"H{random.randint(0, 47)}"  # Random PS node from H0 to H47
    num_workers = 5  # Random number of workers (from 2 to 5 workers)
    worker_nodes = [f"H{random.randint(0, 47)}" for _ in range(num_workers) if ps_node != f"H{random.randint(0, 47)}"]  # Ensure workers are different from PS
    # ps_node = "H0"
    # worker_nodes = [ "H20", "H0"]
    required_bw_per_connection = random.randint(1, 10)

    # Try to allocate bandwidth between the PS and workers with oversubscription enabled
    print(f"\nAttempt {i + 1}: Trying with oversubscription enabled:")
    print(f"PS node: {ps_node}, Worker nodes: {worker_nodes}, Required BW: {required_bw_per_connection} Gbps")
    
    allocation_success = allocate_ps_to_workers(G, ps_node, worker_nodes, required_bw_per_connection, allow_oversubscription=False)

    if allocation_success:
        print(f"Attempt {i + 1}: Bandwidth successfully allocated with oversubscription.")
    else:
        print(f"Attempt {i + 1}: Failed to allocate bandwidth due to insufficient capacity.")

# Plot available bandwidth at each node
plot_node_available_bandwidth(G)

# Plot bandwidth utilization on each link
plot_bandwidth_utilization(G)

# Plot degree distribution of nodes
# plot_degree_distribution(G)

# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10)
# edge_labels = {(u, v): f"{data['reserved_bw']}/{data['bandwidth']} Gbps" for u, v, data in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
# plt.title('Spine-Leaf Topology with Bandwidth Utilization')
# plt.savefig('Spine-Leaf.png')
