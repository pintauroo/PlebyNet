import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

class SpineLeafTopology:
    def __init__(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        self.G = self.create_spine_leaf_topology_with_bandwidth(
            num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine
        )
        self.allocated_paths = {}  # Store paths used for allocation
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        
    def create_spine_leaf_topology_with_bandwidth(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
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
        
    def format_node_ids(self, node_ids):
        # Ensure node_ids is a list of strings prefixed with 'H'
        return [f"H{str(node_id)}" for node_id in node_ids]
    
    def allocate_ps_to_workers(self, ps_node, worker_nodes, required_bw, allow_oversubscription=False):
        """
        Allocate bandwidth between a parameter server (PS) and a list of worker nodes.
        Prevent allocation if bandwidth does not fit and oversubscription is disabled.
        Update bandwidth at the switches only once.
        """


        # Example usage:
        print('ps_node', ps_node)
        ps_node = self.format_node_ids(ps_node)[0]  # Convert 0 to "H0"
        worker_nodes = self.format_node_ids(worker_nodes)  
        paths = {}
        allocated_edges = []  # Track which edges have been allocated to roll back if necessary
        allocated_nodes = set()  # Track which nodes have been updated to avoid double updating

        # First, check if enough bandwidth is available for all workers
        for worker in worker_nodes:
            if ps_node != worker:
                try:
                    path = nx.shortest_path(self.G, source=ps_node, target=worker)
                    paths[worker] = path  # Store the path for this worker
                    print(f"Checking path from PS {ps_node} to worker {worker}: {path}")

                    # Check bandwidth availability on each link in the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        available_bw = self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']
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
                self.G[u][v]['reserved_bw'] += required_bw
                allocated_edges.append((u, v))

                # Only update node reserved_bw once per node, and only if it hasn't been updated already
                if u not in allocated_nodes:
                    self.G.nodes[u]['reserved_bw'] += required_bw
                    allocated_nodes.add(u)
                if v not in allocated_nodes:
                    self.G.nodes[v]['reserved_bw'] += required_bw
                    allocated_nodes.add(v)

                remaining_bw = self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']
                if remaining_bw < 0:
                    if allow_oversubscription:
                        print(f"Allocated with oversubscription. Remaining bandwidth is negative: {remaining_bw} Gbps on link {u}-{v}")
                    else:
                        print(f"Allocation failed: insufficient bandwidth on link {u}-{v} without oversubscription")
                        # Rollback any allocations made before this failure
                        for edge in allocated_edges:
                            u, v = edge
                            self.G[u][v]['reserved_bw'] -= required_bw
                        for node in allocated_nodes:
                            self.G.nodes[node]['reserved_bw'] -= required_bw
                        return False

        # Store the allocated paths for later deallocation
        self.allocated_paths[(ps_node, tuple(worker_nodes))] = paths
        print('Allocation successful!')
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        return True  # Allocation successful

    def deallocate_ps_from_workers(self, ps_node, worker_nodes, required_bw):
        """
        Deallocate the bandwidth that was previously allocated between the PS and the workers.
        It reverses the bandwidth reservation along the stored paths.
        """
        ps_node = self.format_node_ids(ps_node)[0]  # Convert 0 to "H0"
        worker_nodes = self.format_node_ids(worker_nodes)  
        key = (ps_node, tuple(worker_nodes))
        if key not in self.allocated_paths:
            print(f"No allocation found for PS {ps_node} to workers {worker_nodes}.")
            return False  # No allocation to deallocate

        # Get the stored paths for this allocation
        paths = self.allocated_paths[key]

        for worker, path in paths.items():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.G[u][v]['reserved_bw'] -= required_bw
                self.G[u][v]['reserved_bw'] = max(self.G[u][v]['reserved_bw'], 0)  # Ensure no negative bandwidth

                # Only update node reserved_bw once per node, and only if it hasn't been updated already
                self.G.nodes[u]['reserved_bw'] -= required_bw
                self.G.nodes[v]['reserved_bw'] -= required_bw
                self.G.nodes[u]['reserved_bw'] = max(self.G.nodes[u]['reserved_bw'], 0)
                self.G.nodes[v]['reserved_bw'] = max(self.G.nodes[v]['reserved_bw'], 0)

        # Remove the stored paths after deallocation
        del self.allocated_paths[key]
        print(f"Deallocation successful for PS {ps_node} and workers {worker_nodes}!")
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        return True

    def calculate_host_to_host_adjacency_matrix(self):
        """
        Calculate the adjacency matrix that tracks the path capacity (minimum available bandwidth) between host-to-host pairs.
        The path capacity between two hosts is the minimum available bandwidth across the shortest path between them.
        """
        # Get all host nodes
        hosts = [node for node, data in self.G.nodes(data=True) if data['type'] == 'host']
        num_hosts = len(hosts)
        host_to_idx = {host: idx for idx, host in enumerate(hosts)}

        # Initialize an adjacency matrix with 0 (no connection)
        adj_matrix = np.zeros((num_hosts, num_hosts))

        # Iterate through all pairs of host nodes and calculate path capacity
        for i, src in enumerate(hosts):
            for j, dst in enumerate(hosts):
                if src != dst:
                    try:
                        # Get the shortest path between src and dst
                        path = nx.shortest_path(self.G, source=src, target=dst)

                        # Find the minimum available bandwidth (bottleneck) along the path
                        min_available_bw = float('inf')  # Start with a large number
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            available_bw = self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']
                            min_available_bw = min(min_available_bw, available_bw)

                        # Fill the adjacency matrix with the path capacity (bottleneck link)
                        adj_matrix[i][j] = min_available_bw
                        # print(f"Path {src} -> {dst}: Capacity = {min_available_bw} Gbps")

                    except nx.NetworkXNoPath:
                        # If there's no path, leave the entry as 0 (no connection)
                        # print(f"No path between {src} and {dst}")
                        adj_matrix[i][j] = 0

        return adj_matrix

    def plot_node_available_bandwidth(self):
        """
        Plots the available bandwidth at each spine and leaf switch.
        """
        nodes = []
        available_bw = []

        for node, data in self.G.nodes(data=True):
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
        plt.close()


    def plot_bandwidth_utilization(self):
        # Collect bandwidth utilization data for each link
        links = []
        utilizations = []

        for u, v, data in self.G.edges(data=True):
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
        plt.close()

# Example usage
if __name__ == "__main__":
    # Parameters for Medium Deployment
    num_spine_switches = 4  # Typically more spines for higher redundancy
    num_leaf_switches = 6
    num_hosts_per_leaf = 8  # 6 leaves * 8 hosts = 48 servers

    # Bandwidth values in Gbps
    spine_bandwidth = 100  # Uplinks to spine (100 Gbps)
    leaf_bandwidth = 400   # Leaf switches with 4x 100 Gbps uplinks = 400 Gbps
    link_bw_leaf_to_node = 100  # 25 Gbps from leaf to node
    link_bw_leaf_to_spine = 100  # 100 Gbps uplinks to spine

    # Instantiate the topology
    topology = SpineLeafTopology(
        num_spine_switches, num_leaf_switches, num_hosts_per_leaf,
        spine_bandwidth, leaf_bandwidth, link_bw_leaf_to_node, link_bw_leaf_to_spine
    )

    # Example of using the class to allocate bandwidth
    ps_node = "0"
    worker_nodes = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    topology.allocate_ps_to_workers(ps_node, worker_nodes, 10)

    # Deallocate bandwidth after usage
    # topology.deallocate_ps_from_workers(ps_node, worker_nodes, 50)

    # Get adjacency matrix of available bandwidth
    adj_matrix = topology.calculate_host_to_host_adjacency_matrix()
    print("Adjacency Matrix:", adj_matrix)

    # Plot available bandwidth and utilization
    topology.plot_node_available_bandwidth()
    topology.plot_bandwidth_utilization()
