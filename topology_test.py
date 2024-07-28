from src.topology import Topology

def calculate_bandwidth(topology, jobs):
    # Initialize variables
    initialA = topology.get_initial_bw_matrix()# Copy of initial adjacency matrix
    n = len(initialA)  # Number of nodes
    
    # Function to update the adjacency matrix for a job allocation
    
    # Calculate initial total bandwidth (sum of max bandwidth for each node)
    initial_total_bandwidth = topology.get_total_initial_bw()
    
    for i in range(n):
        print(topology.get_updated_bw_matrix()[i])

    print()
    # Allocate jobs
    for job in jobs:
        i, j, bandwidth = job
        topology.allocate_bandwidth(i, j, bandwidth)
    
    # Calculate total available bandwidth (sum of max bandwidth for each node)
    total_available_bandwidth = topology.get_total_remaining_bw()
    
    
    for i in range(n):
        print(topology.get_updated_bw_matrix()[i])

    
    # Calculate total allocated bandwidth
    total_allocated_bandwidth = initial_total_bandwidth - total_available_bandwidth
    
    # Calculate percentage of used bandwidth
    percentage_used_bandwidth = (total_allocated_bandwidth / initial_total_bandwidth) * 100
    
    return initial_total_bandwidth, total_available_bandwidth, total_allocated_bandwidth, percentage_used_bandwidth

# # Example usage
# A = [
#     [0, 100, 100, 100, 100],
#     [100, 0, 100, 100, 100],
#     [100, 100, 0, 100, 100],
#     [100, 100, 100, 0, 100],
#     [100, 100, 100, 100, 0]
# ]


topology = Topology('compute_probabilistic_graph', 100, 10, 0.5)
# A = topology.get_updated_bw_matrix()

# Jobs: (node_i, node_j, bandwidth)
jobs = [
    (1, 2, 100),  # Job between node 1 and node 2 with 100 units of bandwidth
    # (0, 4, 100),  # Job between node 1 and node 2 with 100 units of bandwidth
]

try:
    initial_bandwidth, available_bandwidth, allocated_bandwidth, percentage_used_bandwidth = calculate_bandwidth(topology, jobs)
    print("Initial Total Bandwidth:", initial_bandwidth)
    print("Total Available Bandwidth:", available_bandwidth)
    print("Total Allocated Bandwidth:", allocated_bandwidth)
    print("Percentage of Used Bandwidth:", percentage_used_bandwidth)
except ValueError as e:
    print(e)
