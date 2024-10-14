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
        # Store the original reserved bandwidth for future reference
        self.original_reserved_bw_nodes = {node: data['reserved_bw'] for node, data in self.G.nodes(data=True)}
        self.original_reserved_bw_edges = {(u, v): data['reserved_bw'] for u, v, data in self.G.edges(data=True)}

        

    def create_spine_leaf_topology_with_bandwidth(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        G = nx.Graph()

        # Add spine switches with bandwidth capacity
        spine_switches = [f"S{i}" for i in range(num_spine)]
        for spine in spine_switches:
            G.add_node(spine, type='spine', bandwidth=spine_bw, reserved_bw=0)

        # Add leaf switches with increasing bandwidth capacity
        leaf_switches = [f"L{i}" for i in range(num_leaf)]
        leaf_bw_seq = leaf_bw
        for leaf in leaf_switches:
            G.add_node(leaf, type='leaf', bandwidth=leaf_bw_seq, reserved_bw=0)
            # leaf_bw_seq += 10  # Increment after assignment to start at 10

        # Connect each leaf switch to all spine switches and assign link bandwidth with reserved_bw initialized to 0
        for leaf in leaf_switches:
            for spine in spine_switches:
                G.add_edge(leaf, spine, bandwidth=link_bw_leaf_to_spine, reserved_bw=0)

        # Add hosts and connect them to leaf switches, assigning link bandwidth for each host connection
        host_id = 0
        for leaf in leaf_switches:
            for _ in range(num_hosts_per_leaf):
                host = f"H{host_id}"
                G.add_node(host, type='host', bandwidth=link_bw_leaf_to_node, reserved_bw=0)  # Assign default bandwidth for host nodes
                G.add_edge(leaf, host, bandwidth=link_bw_leaf_to_node, reserved_bw=0)  # Link bw from host to leaf
                host_id += 1

        return G



    def assert_original_state(self):
        """
        Asserts that the network is back to its original state by verifying that
        all nodes and edges have their reserved bandwidth reset to the initial values.
        """
        # Check nodes
        for node, data in self.G.nodes(data=True):
            initial_bw = self.original_reserved_bw_nodes.get(node, 0)
            current_bw = data.get('reserved_bw', 0)
            assert current_bw == initial_bw, (
                f"Node {node} reserved_bw is {current_bw} Gbps, expected {initial_bw} Gbps."
            )

        # Check edges
        for u, v, data in self.G.edges(data=True):
            # Use a consistent representation for edges
            edge = (u, v) if (u, v) in self.original_reserved_bw_edges else (v, u)
            initial_bw = self.original_reserved_bw_edges.get(edge, 0)
            current_bw = data.get('reserved_bw', 0)
            assert current_bw == initial_bw, (
                f"Edge {u}-{v} reserved_bw is {current_bw} Gbps, expected {initial_bw} Gbps."
            )

        print("Assertion passed: Network is back to the original state.")

    def format_node_ids(self, node_ids):
        # Ensure node_ids is a list of strings prefixed with 'H'
        return [f"H{str(node_id)}" for node_id in node_ids]
    
    def allocate_ps_to_workers_balanced(self, worker_nodes, required_bw, allow_oversubscription=False):
        # Format node IDs to ensure consistency
        worker_nodes_formatted = self.format_node_ids(worker_nodes)
        # Get the set of unique PS nodes
        ps_nodes_formatted = set(worker_nodes_formatted)

        # Initialize variables
        paths = {}
        ps_worker_pairs = set()

        # Create unique (PS, worker) pairs
        for ps_node in ps_nodes_formatted:
            for worker_node in worker_nodes_formatted:
                if worker_node != ps_node:
                    ps_worker_pairs.add((ps_node, worker_node))

        # Temporary variables for tentative reservations
        tentative_reserved_bw_nodes = {}
        tentative_reserved_bw_edges = {}

        # Prepare the graph for path finding with edge weights based on available bandwidth
        for u, v, data in self.G.edges(data=True):
            # Calculate available bandwidth
            existing_reserved_bw = data['reserved_bw']
            available_bw = data['bandwidth'] - existing_reserved_bw
            # Assign weight inversely proportional to available bandwidth
            if available_bw > 0:
                data['weight'] = 1 / available_bw
            else:
                data['weight'] = float('inf')  # Avoid edges with no available bandwidth

        # Check bandwidth availability
        for (ps_node, worker_node) in ps_worker_pairs:
            try:
                # Find the shortest path considering edge weights (available bandwidth)
                path = nx.shortest_path(self.G, source=ps_node, target=worker_node, weight='weight')
                paths[(ps_node, worker_node)] = path

                # Check node-level bandwidth
                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        # Total bandwidth after tentative reservation
                        existing_reserved_bw = self.G.nodes[node]['reserved_bw']
                        tentative_bw = tentative_reserved_bw_nodes.get(node, 0)
                        total_reserved_bw = existing_reserved_bw + tentative_bw + required_bw
                        # Check capacity
                        if total_reserved_bw > self.G.nodes[node]['bandwidth']:
                            print(
                                f"Insufficient bandwidth on node {node}. "
                                f"Cannot allocate {required_bw} Gbps."
                            )
                            return False  # Allocation cannot proceed
                        # Tentatively reserve bandwidth
                        tentative_reserved_bw_nodes[node] = tentative_bw + required_bw

                # Check edge-level bandwidth
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # Use a consistent representation for edges
                    edge = frozenset([u, v])  # Edge represented as an unordered pair
                    existing_reserved_bw = self.G[u][v]['reserved_bw']
                    tentative_bw = tentative_reserved_bw_edges.get(edge, 0)
                    total_reserved_bw = existing_reserved_bw + tentative_bw + required_bw
                    # Check capacity
                    if total_reserved_bw > self.G[u][v]['bandwidth']:
                        print(
                            f"Insufficient bandwidth on link {u}-{v}. "
                            f"Cannot allocate {required_bw} Gbps."
                        )
                        return False  # Allocation cannot proceed
                    # Tentatively reserve bandwidth
                    tentative_reserved_bw_edges[edge] = tentative_bw + required_bw

            except nx.NetworkXNoPath:
                print(f"No path found between PS {ps_node} and worker {worker_node}.")
                return False  # Allocation cannot proceed

        # All checks passed; finalize reservations
        for node, bw in tentative_reserved_bw_nodes.items():
            self.G.nodes[node]['reserved_bw'] += bw
            # Assert after reservation
            assert self.G.nodes[node]['reserved_bw'] <= self.G.nodes[node]['bandwidth'], (
                f"Node {node} over-allocated after reservation. "
                f"Reserved: {self.G.nodes[node]['reserved_bw']} Gbps, "
                f"Capacity: {self.G.nodes[node]['bandwidth']} Gbps"
            )

        for edge, bw in tentative_reserved_bw_edges.items():
            u, v = tuple(edge)
            self.G[u][v]['reserved_bw'] += bw
            # Assert after reservation
            assert self.G[u][v]['reserved_bw'] <= self.G[u][v]['bandwidth'], (
                f"Edge {u}-{v} over-allocated after reservation. "
                f"Reserved: {self.G[u][v]['reserved_bw']} Gbps, "
                f"Capacity: {self.G[u][v]['bandwidth']} Gbps"
            )

        # Store allocated paths
        self.allocated_paths.update(paths)

        # Assert that all paths are valid
        for (ps_node, worker_node), path in self.allocated_paths.items():
            assert nx.has_path(self.G, ps_node, worker_node), (
                f"No valid path stored for {ps_node} to {worker_node}."
            )
            # Verify that the path connects the correct nodes
            assert path[0] == ps_node and path[-1] == worker_node, (
                f"Path does not connect {ps_node} to {worker_node}."
            )

        self.adj = self.calculate_host_to_host_adjacency_matrix()

        # Final assertion to ensure overall consistency
        for node in self.G.nodes():
            if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                assert self.G.nodes[node]['reserved_bw'] <= self.G.nodes[node]['bandwidth'], (
                    f"Node {node} over-allocated in final check. "
                    f"Reserved: {self.G.nodes[node]['reserved_bw']} Gbps, "
                    f"Capacity: {self.G.nodes[node]['bandwidth']} Gbps"
                )

        for u, v in self.G.edges():
            assert self.G[u][v]['reserved_bw'] <= self.G[u][v]['bandwidth'], (
                f"Edge {u}-{v} over-allocated in final check. "
                f"Reserved: {self.G[u][v]['reserved_bw']} Gbps, "
                f"Capacity: {self.G[u][v]['bandwidth']} Gbps"
            )

        return True




    def allocate_ps_to_workers_single(self, ps_node, worker_nodes, required_bw, allow_oversubscription=False):
        """
        Allocate bandwidth between a parameter server (PS) and a list of worker nodes.
        Reduce bandwidth for each worker independently, even if they share paths through the same switch.
        """
        ps_node = self.format_node_ids(ps_node)[0]  # Correctly format ps_node
        worker_nodes = self.format_node_ids(worker_nodes)
        paths = {}
        cumulative_node_bw = {}
        cumulative_edge_bw = {}
        allocated_nodes = set()
        allocated_edges = []
        modifications = []

        # First, check cumulative bandwidth requirements
        for worker in worker_nodes:
            if ps_node == worker:
                continue  # Skip allocation if PS and worker are the same node

            try:
                path = nx.shortest_path(self.G, source=ps_node, target=worker)
                paths[worker] = path

                # Accumulate node-level bandwidth requirements
                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        cumulative_node_bw[node] = cumulative_node_bw.get(node, 0) + required_bw
                        total_reserved_bw = self.G.nodes[node]['reserved_bw'] + cumulative_node_bw[node]
                        if total_reserved_bw > self.G.nodes[node]['bandwidth']:
                            print(
                                f"Not enough bandwidth on node {node}. "
                                f"Available: {self.G.nodes[node]['bandwidth'] - self.G.nodes[node]['reserved_bw']} Gbps, "
                                f"Required: {cumulative_node_bw[node]} Gbps"
                            )
                            return False

                # Accumulate edge-level bandwidth requirements
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = (u, v) if (u, v) in self.G.edges else (v, u)
                    cumulative_edge_bw[edge] = cumulative_edge_bw.get(edge, 0) + required_bw
                    total_reserved_bw = self.G[u][v]['reserved_bw'] + cumulative_edge_bw[edge]
                    if total_reserved_bw > self.G[u][v]['bandwidth']:
                        if not allow_oversubscription:
                            print(
                                f"Not enough bandwidth on link {u}-{v}. "
                                f"Available: {self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']} Gbps, "
                                f"Required: {cumulative_edge_bw[edge]} Gbps"
                            )
                            return False

            except nx.NetworkXNoPath:
                print(f"No path found between PS {ps_node} and worker {worker}.")
                return False

        # If bandwidth is available, reserve it
        try:
            for worker in worker_nodes:
                if ps_node == worker:
                    continue  # Skip reservation if PS and worker are the same node

                # Reserve bandwidth at the switches (spine/leaf switches in the path)
                for node in paths[worker]:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        new_reserved_bw = self.G.nodes[node]['reserved_bw'] + required_bw
                        if new_reserved_bw > self.G.nodes[node]['bandwidth']:
                            raise Exception(f"Not enough bandwidth on node {node} during allocation.")
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] = new_reserved_bw
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))
                        allocated_nodes.add(node)

                # Reserve bandwidth at the edges (links)
                for i in range(len(paths[worker]) - 1):
                    u, v = paths[worker][i], paths[worker][i + 1]
                    new_reserved_bw = self.G[u][v]['reserved_bw'] + required_bw
                    if new_reserved_bw > self.G[u][v]['bandwidth']:
                        if not allow_oversubscription:
                            raise Exception(f"Not enough bandwidth on link {u}-{v} during allocation.")
                    old_reserved_bw = self.G[u][v]['reserved_bw']
                    self.G[u][v]['reserved_bw'] = new_reserved_bw
                    modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))
                    allocated_edges.append((u, v))

            # Allocation successful
            for worker_node in worker_nodes:
                if ps_node == worker_node:
                    continue
                self.allocated_paths[(ps_node, worker_node)] = paths[worker_node]

            self.adj = self.calculate_host_to_host_adjacency_matrix()
            return True
        except Exception as e:
            print(f"Exception occurred during allocation: {e}")
            # Rollback modifications
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return False





    def deallocate_ps_from_workers(self, worker_nodes, required_bw):
        """
        Deallocate the bandwidth that was previously allocated between the PS and the workers.
        It reverses the bandwidth reservation along the stored paths.
        """
        # Format node IDs to ensure consistency
        worker_nodes_formatted = self.format_node_ids(worker_nodes)
        # Get the set of unique PS nodes
        ps_nodes_formatted = set(worker_nodes_formatted)

        rollback_required = False

        # Store initial reserved bandwidth for rollback purposes
        initial_node_reserved_bw = {}
        initial_edge_reserved_bw = {}

        # Check if any allocations exist between PS and workers
        paths_to_remove = []
        for (ps_node, worker_node), path in self.allocated_paths.items():
            if ps_node in ps_nodes_formatted and worker_node in worker_nodes_formatted:
                # 1. Deallocate bandwidth at the switches (spine and leaf switches)
                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        initial_node_reserved_bw[node] = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] -= required_bw
                        self.G.nodes[node]['reserved_bw'] = max(self.G.nodes[node]['reserved_bw'], 0)

                # 2. Deallocate bandwidth at the edges (links)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = (u, v) if (u, v) in self.G.edges else (v, u)
                    initial_edge_reserved_bw[edge] = self.G[u][v]['reserved_bw']
                    self.G[u][v]['reserved_bw'] -= required_bw
                    self.G[u][v]['reserved_bw'] = max(self.G[u][v]['reserved_bw'], 0)

                # Mark the path for removal after deallocation
                paths_to_remove.append((ps_node, worker_node))

        # Remove the stored paths after deallocation
        for key in paths_to_remove:
            del self.allocated_paths[key]

        # Recalculate the adjacency matrix after deallocation
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        return True




    def calculate_host_to_host_adjacency_matrix(self):
        """
        Calculate the adjacency matrix that tracks the path capacity (minimum available bandwidth)
        between host-to-host pairs. The path capacity between two hosts is the maximum of the
        minimum available bandwidth across all shortest paths between them, considering both
        edge (link) and node (switch) capacities.

        Returns:
        - adj_matrix (numpy.ndarray): A 2D array representing the adjacency matrix.
        """
        # Identify all host nodes
        hosts = [node for node, data in self.G.nodes(data=True) if data.get('type') == 'host']
        num_hosts = len(hosts)
        host_to_idx = {host: idx for idx, host in enumerate(hosts)}

        # Initialize adjacency matrix with zeros
        adj_matrix = np.zeros((num_hosts, num_hosts))

        # Iterate through all unique host pairs
        for i, src in enumerate(hosts):
            for j, dst in enumerate(hosts):
                if src == dst:
                    continue  # Skip same host

                try:
                    # Retrieve all shortest paths between src and dst
                    paths = list(nx.all_shortest_paths(self.G, source=src, target=dst))

                    max_min_bw = 0  # Initialize maximum of minimum bandwidths

                    for path in paths:
                        min_available_bw = float('inf')  # Initialize minimum available bandwidth for the path
                        valid_path = True  # Flag to determine if the path is valid

                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            edge_data = self.G.get_edge_data(u, v)

                            # Check if edge data exists and has necessary attributes
                            if edge_data is None:
                                valid_path = False
                                break

                            bandwidth = edge_data.get('bandwidth')
                            reserved_bw = edge_data.get('reserved_bw')

                            if bandwidth is None or reserved_bw is None:
                                valid_path = False
                                break

                            available_bw_edge = max(bandwidth - reserved_bw, 0)

                            # Determine if node 'v' is an intermediate switch (not a host)
                            node_v_data = self.G.nodes[v]
                            if node_v_data.get('type') != 'host':
                                node_bandwidth = node_v_data.get('bandwidth', float('inf'))
                                node_reserved_bw = node_v_data.get('reserved_bw', 0)
                                available_bw_node = max(node_bandwidth - node_reserved_bw, 0)
                            else:
                                # Hosts are considered to have no bandwidth constraints in this context
                                available_bw_node = float('inf')

                            # The available bandwidth for this segment is the minimum of edge and node bandwidth
                            available_bw = min(available_bw_edge, available_bw_node)

                            # Update the minimum available bandwidth for the entire path
                            min_available_bw = min(min_available_bw, available_bw)

                        if valid_path:
                            # Update the maximum of minimum bandwidths across all paths
                            max_min_bw = max(max_min_bw, min_available_bw)

                    # Assign the calculated bottleneck bandwidth to the adjacency matrix
                    adj_matrix[i][j] = max_min_bw

                except nx.NetworkXNoPath:
                    # If no path exists, the bandwidth remains zero
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
        # Collect bandwidth utilization data for links connected to host nodes
        links = []
        utilizations = []

        for u, v, data in self.G.edges(data=True):
            # Check if either node u or node v is a host node
            u_type = self.G.nodes[u].get('type')
            v_type = self.G.nodes[v].get('type')

            if u_type == 'host' or v_type == 'host':
                total_bw = data['bandwidth']
                reserved_bw = data['reserved_bw']
                utilization = total_bw - reserved_bw
                links.append(f"{u}-{v}")
                utilizations.append(utilization)

        # Plotting bandwidth utilization as a bar chart
        plt.figure(figsize=(20, 6))
        plt.bar(links, utilizations, color='skyblue')
        plt.xlabel('Links')
        plt.ylabel('Bandwidth Utilization')
        plt.title('Bandwidth Utilization on Links Connected to Host Nodes')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig('plot_bandwidth_utilization_hosts.png')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Parameters for Medium Deployment
    num_spine_switches = 4  # Typically more spines for higher redundancy
    num_leaf_switches = 6
    num_hosts_per_leaf = 8  # 6 leaves * 8 hosts = 48 servers

    # Bandwidth values in Gbps
    spine_bandwidth = 200  # Uplinks to spine (100 Gbps)
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
    # worker_nodes = ["10","20","30","40"]
    worker_nodes = ["10","20"]
    topology.allocate_ps_to_workers(ps_node, worker_nodes, 25)



    # Deallocate bandwidth after usage
    # topology.deallocate_ps_from_workers(ps_node, worker_nodes, 50)

    # Get adjacency matrix of available bandwidth
    adj_matrix = topology.calculate_host_to_host_adjacency_matrix()
    # print("Adjacency Matrix:", adj_matrix)

    # Plot available bandwidth and utilization
    topology.plot_node_available_bandwidth()
    topology.plot_bandwidth_utilization()
