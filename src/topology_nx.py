import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class BaseTopology:
    def __init__(self):
        self.G = nx.Graph()
        self.allocated_paths = {}  # key: job_id, value: dict of allocations
        self.utilization_records = []
        self.allocation_step = 0

    def create_topology(self):
        """
        Abstract method to create a topology. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_host_to_host_adjacency_matrix(self):
        hosts = [node for node, data in self.G.nodes(data=True) if data.get('type') == 'host']
        num_hosts = len(hosts)
        host_to_idx = {host: idx for idx, host in enumerate(hosts)}

        adj_matrix = np.zeros((num_hosts, num_hosts))

        for i, src in enumerate(hosts):
            for j, dst in enumerate(hosts):
                if src == dst:
                    continue

                try:
                    paths = list(nx.all_shortest_paths(self.G, source=src, target=dst))
                    max_min_bw = 0

                    for path in paths:
                        min_available_bw = float('inf')
                        valid_path = True

                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            edge_data = self.G.get_edge_data(u, v)

                            if edge_data is None:
                                valid_path = False
                                break

                            bandwidth = edge_data.get('bandwidth')
                            reserved_bw = edge_data.get('reserved_bw')

                            if bandwidth is None or reserved_bw is None:
                                valid_path = False
                                break

                            available_bw_edge = max(bandwidth - reserved_bw, 0)
                            node_v_data = self.G.nodes[v]
                            if node_v_data.get('type') != 'host':
                                node_bandwidth = node_v_data.get('bandwidth', float('inf'))
                                node_reserved_bw = node_v_data.get('reserved_bw', 0)
                                available_bw_node = max(node_bandwidth - node_reserved_bw, 0)
                            else:
                                available_bw_node = float('inf')

                            available_bw = min(available_bw_edge, available_bw_node)
                            min_available_bw = min(min_available_bw, available_bw)

                        if valid_path:
                            max_min_bw = max(max_min_bw, min_available_bw)

                    adj_matrix[i][j] = max_min_bw

                except nx.NetworkXNoPath:
                    adj_matrix[i][j] = 0

        return adj_matrix

    def record_utilization(self):
        record = {'allocation_step': self.allocation_step}
        
        for node, data in self.G.nodes(data=True):
            if data['type'] in ['spine', 'leaf', 'host']:
                prefix = 'node'
                element = node
                record[f"{prefix}_{element}_total_bw"] = data['bandwidth']
                record[f"{prefix}_{element}_reserved_bw"] = data['reserved_bw']
                record[f"{prefix}_{element}_available_bw"] = data['bandwidth'] - data['reserved_bw']

        for u, v, data in self.G.edges(data=True):
            prefix = 'edge'
            element = f"{u}-{v}"
            record[f"{prefix}_{element}_total_bw"] = data['bandwidth']
            record[f"{prefix}_{element}_reserved_bw"] = data['reserved_bw']
            record[f"{prefix}_{element}_available_bw"] = data['bandwidth'] - data['reserved_bw']
        
        self.utilization_records.append(record)
        self.allocation_step += 1

    def save_stats_to_csv(self, filename):
        if not filename.endswith('.csv'):
            filename += '.csv'
        df = pd.DataFrame(self.utilization_records)
        df.to_csv(filename, index=False)
        # print(f"Utilization statistics saved to {filename}")

    def get_utilization_percentage(self):
        """
        Calculate the utilization percentage for each node and edge in the network,
        generate visualizations, and save them as image files.

        Returns:
            dict: A dictionary containing two dictionaries:
                  - 'nodes': Utilization percentages for each node.
                  - 'edges': Utilization percentages for each edge.
        """
        # --------------------------
        # 1. Calculate Node Utilization
        # --------------------------
        node_utilization = {}
        for node, data in self.G.nodes(data=True):
            bandwidth = data.get('bandwidth', 0)
            reserved_bw = data.get('reserved_bw', 0)
            if bandwidth > 0:
                utilization = (reserved_bw / bandwidth) * 100
            else:
                utilization = 0
            node_utilization[node] = float(round(utilization, 2))  # Ensure Python float

        # --------------------------
        # 2. Calculate Edge Utilization
        # --------------------------
        edge_utilization = {}
        for u, v, data in self.G.edges(data=True):
            bandwidth = data.get('bandwidth', 0)
            reserved_bw = data.get('reserved_bw', 0)
            if bandwidth > 0:
                utilization = (reserved_bw / bandwidth) * 100
            else:
                utilization = 0
            # Convert tuple to string for easier plotting
            edge_str = f"{u}-{v}"
            edge_utilization[edge_str] = float(round(utilization, 2))  # Ensure Python float

        # --------------------------
        # 3. Compile Utilization Data
        # --------------------------
        utilization = {'nodes': node_utilization, 'edges': edge_utilization}

        # --------------------------
        # 4. Convert Edge Utilization to DataFrame
        # --------------------------
        edges_df = pd.DataFrame(list(edge_utilization.items()), columns=['Edge', 'Utilization (%)'])

        # Split 'Edge' into 'Source' and 'Destination'
        edges_df[['Source', 'Destination']] = edges_df['Edge'].str.split('-', expand=True)

        # --------------------------
        # 5. Pivot Data for Heatmap
        # --------------------------
        try:
            heatmap_data = edges_df.pivot(index="Source", columns="Destination", values="Utilization (%)")
        except ValueError as e:
            # print(f"Error during pivot operation: {e}")
            # print("Inspecting the 'edges_df' DataFrame:")
            # print(edges_df.head())
            raise

        # --------------------------
        # 6. Visualization: Edge Utilization Heatmap
        # --------------------------
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
        plt.title('Edge Utilization Heatmap')
        plt.xlabel('Destination Node')
        plt.ylabel('Source Node')
        plt.tight_layout()
        plt.savefig('BWUTIL_heatmap.png')
        plt.close()

        # --------------------------
        # 7. Visualization: Node Utilization Bar Chart
        # --------------------------
        nodes_df = pd.DataFrame(list(node_utilization.items()), columns=['Node', 'Utilization (%)'])

        plt.figure(figsize=(14, 7))
        sns.barplot(x='Node', y='Utilization (%)', data=nodes_df)
        plt.title('Node Utilization Percentage')
        plt.xlabel('Node')
        plt.ylabel('Utilization (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Node_Utilization_BarChart.png')
        plt.close()

        # --------------------------
        # 8. Visualization: Edge Utilization Bar Chart
        # --------------------------
        plt.figure(figsize=(14, 7))
        sns.barplot(x='Edge', y='Utilization (%)', data=edges_df)
        plt.title('Edge Utilization Percentage')
        plt.xlabel('Edge')
        plt.ylabel('Utilization (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('Edge_Utilization_BarChart.png')
        plt.close()

        # --------------------------
        # 9. Visualization: Sorted Node Utilization Bar Chart
        # --------------------------
        sorted_nodes_df = nodes_df.sort_values(by='Utilization (%)', ascending=False)

        plt.figure(figsize=(14, 7))
        sns.barplot(x='Node', y='Utilization (%)', data=sorted_nodes_df)
        plt.title('Sorted Node Utilization Percentage')
        plt.xlabel('Node')
        plt.ylabel('Utilization (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Sorted_Node_Utilization_BarChart.png')
        plt.close()

        # --------------------------
        # 10. Visualization: Sorted Edge Utilization Bar Chart
        # --------------------------
        sorted_edges_df = edges_df.sort_values(by='Utilization (%)', ascending=False)

        plt.figure(figsize=(14, 7))
        sns.barplot(x='Edge', y='Utilization (%)', data=sorted_edges_df)
        plt.title('Sorted Edge Utilization Percentage')
        plt.xlabel('Edge')
        plt.ylabel('Utilization (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('Sorted_Edge_Utilization_BarChart.png')
        plt.close()

        return utilization
    
    def allocate_ps_to_workers_balanced(self, worker_nodes, required_bw, job_id, allow_oversubscription=False):
        # Format node IDs
        worker_nodes_formatted = self.format_node_ids(worker_nodes)
        if not worker_nodes_formatted:
            # print("No worker nodes provided for allocation.")
            return 0

        # In this context, PS nodes are the unique set of worker nodes
        ps_nodes_formatted = list(set(worker_nodes_formatted))
        paths = {}
        edge_usage_count = {}
        node_usage_count = {}
        modifications = []

        # Collect all paths and usage counts
        for worker in worker_nodes_formatted:
            for ps in ps_nodes_formatted:
                try:
                    path = nx.shortest_path(self.G, source=ps, target=worker)
                    key = (ps, worker)
                    if key not in paths:
                        paths[key] = {'path': path, 'count': 1}
                    else:
                        paths[key]['count'] += 1

                    # Update node usage counts
                    for node in path:
                        if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                            node_usage_count[node] = node_usage_count.get(node, 0) + 1

                    # Update edge usage counts
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge = (u, v) if (u, v) in self.G.edges else (v, u)
                        edge_usage_count[edge] = edge_usage_count.get(edge, 0) + 1

                except nx.NetworkXNoPath:
                    # print(f"No path found between PS {ps} and worker {worker}.")
                    return 0

        # Compute the maximum per-connection bandwidth
        max_bw_per_connection = float('inf')

        for node, usage in node_usage_count.items():
            available_bw = self.G.nodes[node]['bandwidth'] - self.G.nodes[node]['reserved_bw']
            per_connection_bw = available_bw / usage
            if per_connection_bw < max_bw_per_connection:
                max_bw_per_connection = per_connection_bw

        for edge, usage in edge_usage_count.items():
            u, v = edge
            available_bw = self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']
            per_connection_bw = available_bw / usage
            if per_connection_bw < max_bw_per_connection:
                max_bw_per_connection = per_connection_bw

        # Limit by required_bw if specified
        max_bw_per_connection = min(max_bw_per_connection, required_bw)

        if max_bw_per_connection <= 1e-6:
            # print("No bandwidth available for allocation.")
            return 0

        # Proceed to allocate max_bw_per_connection to each path
        try:
            for (ps, worker), path_info in paths.items():
                path = path_info['path']
                count = path_info['count']
                total_bw = max_bw_per_connection * count

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        new_reserved_bw = self.G.nodes[node]['reserved_bw'] + total_bw
                        if new_reserved_bw > self.G.nodes[node]['bandwidth']:
                            if not allow_oversubscription:
                                raise Exception(f"Not enough bandwidth on node {node} during allocation.")
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] = round(new_reserved_bw, 6)
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    new_reserved_bw = self.G[u][v]['reserved_bw'] + total_bw
                    if new_reserved_bw > self.G[u][v]['bandwidth']:
                        if not allow_oversubscription:
                            raise Exception(f"Not enough bandwidth on link {u}-{v} during allocation.")
                    old_reserved_bw = self.G[u][v]['reserved_bw']
                    self.G[u][v]['reserved_bw'] = round(new_reserved_bw, 6)
                    modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))

            # Record the allocation under the job_id
            if job_id not in self.allocated_paths:
                self.allocated_paths[job_id] = {}

            for (ps, worker), path_info in paths.items():
                self.allocated_paths[job_id][(ps, worker)] = (path_info['path'], max_bw_per_connection, path_info['count'])

            self.record_utilization()
            return max_bw_per_connection

        except Exception as e:
            # print(f"Exception occurred during allocation: {e}")
            # Rollback any changes due to exception
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return 0

    def deallocate_ps_to_workers_balanced(self, job_id):
        """
        Deallocate resources for a job allocated using allocate_ps_to_workers_balanced.
        """
        if job_id not in self.allocated_paths:
            # print(f"No allocations found for job_id {job_id}")
            return False

        allocations = self.allocated_paths[job_id]
        modifications = []
        try:
            for (ps, worker), (path, allocated_bw, count) in allocations.items():
                total_bw = allocated_bw * count

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        new_reserved_bw = self.G.nodes[node]['reserved_bw'] - total_bw
                        self.G.nodes[node]['reserved_bw'] = max(round(new_reserved_bw, 6), 0)
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    old_reserved_bw = self.G[u][v]['reserved_bw']
                    new_reserved_bw = self.G[u][v]['reserved_bw'] - total_bw
                    self.G[u][v]['reserved_bw'] = max(round(new_reserved_bw, 6), 0)
                    modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))

            # After successful deallocation, remove the job_id from allocated_paths
            del self.allocated_paths[job_id]
            self.record_utilization()
            return True
        except Exception as e:
            # print(f"Exception occurred during deallocation: {e}")
            # In case of exception, roll back changes
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return False

    def format_node_ids(self, node_ids):
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        return [f"H{str(node_id)}" for node_id in node_ids]

    def compute_max_allocatable_bw(self, ps_node, worker_nodes, required_bw, max_bw=None, allow_oversubscription=False):
        ps_node = self.format_node_ids([ps_node])[0]
        worker_nodes = self.format_node_ids(worker_nodes)
        paths = {}
        edge_usage_count = {}
        node_usage_count = {}
        modifications = []

        for worker in worker_nodes:
            if ps_node == worker:
                continue

            try:
                path = nx.shortest_path(self.G, source=ps_node, target=worker)
                paths[worker] = path

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        node_usage_count[node] = node_usage_count.get(node, 0) + 1

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = (u, v) if (u, v) in self.G.edges else (v, u)
                    edge_usage_count[edge] = edge_usage_count.get(edge, 0) + 1

            except nx.NetworkXNoPath:
                # print(f"No path found between PS {ps_node} and worker {worker}.")
                return 0  # Allocation is impossible due to no path

        # Compute the maximum per-worker bandwidth
        max_bw_per_worker = float('inf')

        for node, usage in node_usage_count.items():
            available_bw = self.G.nodes[node]['bandwidth'] - self.G.nodes[node]['reserved_bw']
            per_worker_bw = available_bw / usage
            if per_worker_bw < max_bw_per_worker:
                max_bw_per_worker = per_worker_bw

        for edge, usage in edge_usage_count.items():
            u, v = edge
            available_bw = self.G[u][v]['bandwidth'] - self.G[u][v]['reserved_bw']
            per_worker_bw = available_bw / usage
            if per_worker_bw < max_bw_per_worker:
                max_bw_per_worker = per_worker_bw

        if max_bw is not None:
            max_bw_per_worker = min(max_bw_per_worker, max_bw)

        if max_bw_per_worker <= 0:
            # print("No bandwidth available for allocation.")
            return 0

        return max_bw_per_worker

class SpineLeafTopology(BaseTopology):
    def __init__(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        super().__init__()
        self.G = self.create_topology(num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine)
        self.original_reserved_bw_nodes = {node: data['reserved_bw'] for node, data in self.G.nodes(data=True)}
        self.original_reserved_bw_edges = {(u, v): data['reserved_bw'] for u, v, data in self.G.edges(data=True)}
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        

    def create_topology(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        G = nx.Graph()

        spine_switches = [f"S{i}" for i in range(num_spine)]
        for spine in spine_switches:
            G.add_node(spine, type='spine', bandwidth=spine_bw, reserved_bw=0)

        leaf_switches = [f"L{i}" for i in range(num_leaf)]
        for leaf in leaf_switches:
            G.add_node(leaf, type='leaf', bandwidth=leaf_bw, reserved_bw=0)

        for leaf in leaf_switches:
            for spine in spine_switches:
                G.add_edge(leaf, spine, bandwidth=link_bw_leaf_to_spine, reserved_bw=0)

        host_id = 0
        for leaf in leaf_switches:
            for _ in range(num_hosts_per_leaf):
                host = f"H{host_id}"
                G.add_node(host, type='host', bandwidth=link_bw_leaf_to_node, reserved_bw=0)
                G.add_edge(leaf, host, bandwidth=link_bw_leaf_to_node, reserved_bw=0)
                host_id += 1

        return G

    def verify_network_state(self):
        """
        Verify that the network's reserved bandwidths are back to their original state.
        """
        epsilon = 1e-6  # Tolerance level for floating-point comparison
        # Check nodes
        for node, data in self.G.nodes(data=True):
            original_bw = self.original_reserved_bw_nodes.get(node, 0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                # print(f"Node {node} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # Check edges
        for u, v, data in self.G.edges(data=True):
            edge = (u, v)
            original_bw = self.original_reserved_bw_edges.get(edge, 0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                # print(f"Edge {u}-{v} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # print("Network state verification passed. The network is back to its original state.")
        return True

    def allocate_job_max_bandwidth(self, allocation_list, total_required_bw, job_id, allow_oversubscription=False):
        """
        Allocate the maximum possible bandwidth for a job up to the required total bandwidth.

        Parameters:
            allocation_list (list): List of worker node IDs (integers) to which bandwidth is to be allocated.
            total_required_bw (float): Total bandwidth required for the job.
            job_id (int): Unique identifier for the job.
            allow_oversubscription (bool): Whether to allow oversubscription of node capacities.

        Returns:
            tuple:
                allocated_per_connection_bw (float): Bandwidth allocated per connection.
                total_allocated_bw (float): Total bandwidth allocated for the job.
        """
        num_connections = len(allocation_list)
        if num_connections == 0:
            # print("Allocation list is empty.")
            return 0, 0

        required_per_connection = total_required_bw / num_connections
        # print(f"Required bandwidth per connection: {required_per_connection}")

        # Allocate using the existing balanced allocation function
        allocated_per_connection_bw = self.allocate_ps_to_workers_balanced(
            allocation_list, required_per_connection, job_id, allow_oversubscription=allow_oversubscription
        )

        total_allocated_bw = allocated_per_connection_bw * num_connections
        # print(f"Allocated bandwidth per connection: {allocated_per_connection_bw}")
        # print(f"Total allocated bandwidth: {total_allocated_bw}")

        # if total_allocated_bw < total_required_bw:
            # print(f"Warning: Only {total_allocated_bw} out of {total_required_bw} bandwidth could be allocated.")

        return allocated_per_connection_bw, total_allocated_bw

class SpineLeafTopology(BaseTopology):
    def __init__(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        super().__init__()
        self.G = self.create_topology(num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine)
        self.original_reserved_bw_nodes = {node: data['reserved_bw'] for node, data in self.G.nodes(data=True)}
        self.original_reserved_bw_edges = {(u, v): data['reserved_bw'] for u, v, data in self.G.edges(data=True)}
        self.adj = self.calculate_host_to_host_adjacency_matrix()
        

    def create_topology(self, num_spine, num_leaf, num_hosts_per_leaf, spine_bw, leaf_bw, link_bw_leaf_to_node, link_bw_leaf_to_spine):
        G = nx.Graph()

        spine_switches = [f"S{i}" for i in range(num_spine)]
        for spine in spine_switches:
            G.add_node(spine, type='spine', bandwidth=spine_bw, reserved_bw=0)

        leaf_switches = [f"L{i}" for i in range(num_leaf)]
        for leaf in leaf_switches:
            G.add_node(leaf, type='leaf', bandwidth=leaf_bw, reserved_bw=0)

        for leaf in leaf_switches:
            for spine in spine_switches:
                G.add_edge(leaf, spine, bandwidth=link_bw_leaf_to_spine, reserved_bw=0)

        host_id = 0
        for leaf in leaf_switches:
            for _ in range(num_hosts_per_leaf):
                host = f"H{host_id}"
                G.add_node(host, type='host', bandwidth=link_bw_leaf_to_node, reserved_bw=0)
                G.add_edge(leaf, host, bandwidth=link_bw_leaf_to_node, reserved_bw=0)
                host_id += 1

        return G

    def verify_network_state(self):
        """
        Verify that the network's reserved bandwidths are back to their original state.
        """
        epsilon = 1e-6  # Tolerance level for floating-point comparison
        # Check nodes
        for node, data in self.G.nodes(data=True):
            original_bw = self.original_reserved_bw_nodes.get(node, 0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                # print(f"Node {node} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # Check edges
        for u, v, data in self.G.edges(data=True):
            edge = (u, v)
            original_bw = self.original_reserved_bw_edges.get(edge, 0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                # print(f"Edge {u}-{v} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # print("Network state verification passed. The network is back to its original state.")
        return True

    def allocate_job_max_bandwidth(self, allocation_list, total_required_bw, job_id, allow_oversubscription=False):
        """
        Allocate the maximum possible bandwidth for a job up to the required total bandwidth.

        Parameters:
            allocation_list (list): List of worker node IDs (integers) to which bandwidth is to be allocated.
            total_required_bw (float): Total bandwidth required for the job.
            job_id (int): Unique identifier for the job.
            allow_oversubscription (bool): Whether to allow oversubscription of node capacities.

        Returns:
            tuple:
                allocated_per_connection_bw (float): Bandwidth allocated per connection.
                total_allocated_bw (float): Total bandwidth allocated for the job.
        """
        num_connections = len(allocation_list)
        if num_connections == 0:
            # print("Allocation list is empty.")
            return 0, 0

        required_per_connection = total_required_bw / num_connections
        # print(f"Required bandwidth per connection: {required_per_connection}")

        # Allocate using the existing balanced allocation function
        allocated_per_connection_bw = self.allocate_ps_to_workers_balanced(
            allocation_list, required_per_connection, job_id, allow_oversubscription=allow_oversubscription
        )

        total_allocated_bw = allocated_per_connection_bw * num_connections
        # print(f"Allocated bandwidth per connection: {allocated_per_connection_bw}")
        # print(f"Total allocated bandwidth: {total_allocated_bw}")

        # if total_allocated_bw < total_required_bw:
            # print(f"Warning: Only {total_allocated_bw} out of {total_required_bw} bandwidth could be allocated.")

        return allocated_per_connection_bw, total_allocated_bw

# Example usage
if __name__ == "__main__":
    # Parameters based on user-specified configuration
    num_spine_switches = 2
    num_leaf_switches = 5
    num_hosts_per_leaf = 10

    spine_bandwidth = 500  # max_spine_capacity
    leaf_bandwidth = 500   # max_leaf_capacity
    link_bw_leaf_to_node = 100  # max_node_bw
    link_bw_leaf_to_spine = 100  # max_leaf_to_spine_bw

    topology = SpineLeafTopology(
        num_spine_switches, num_leaf_switches, num_hosts_per_leaf,
        spine_bandwidth, leaf_bandwidth, link_bw_leaf_to_node, link_bw_leaf_to_spine
    )

    # print("Adjacency Matrix calculated.")

    # Define the job allocation
    allocation_list = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    # This corresponds to hosts H3, H3, H4, H4, H5, H5, H6, H6, H7, H7
    total_bw = 35  # Total bandwidth required
    job_id = 1

    # Allocate resources for the job
    # Note: Since total available node capacity is 500*2 (spines) + 500*5 (leaves) = 3500,
    # and each connection requires up to 355.1 BW, which exceeds node capacities.
    # Therefore, the allocation will be capped at the maximum possible without oversubscription.
    # To strictly enforce not exceeding capacities, set allow_oversubscription=False
    allocated_per_conn, total_allocated = topology.allocate_job_max_bandwidth(
        allocation_list, total_bw, job_id, allow_oversubscription=False
    )
    # print(f"Allocation success for job {job_id}: {allocated_per_conn > 0}")
    # print(f"Allocated bandwidth per connection for job {job_id}: {allocated_per_conn}")
    # print(f"Total allocated bandwidth for job {job_id}: {total_allocated}")

    # Display current allocated paths
    # print(f"Current allocated paths: {topology.allocated_paths}")

    # Save utilization statistics after allocation
    topology.save_stats_to_csv('topology_utilization_after_allocation.csv')

    # Deallocate resources for the job
    deallocation_success = topology.deallocate_ps_to_workers_balanced(job_id)
    # print(f"Deallocation success for job {job_id}: {deallocation_success}")

    # Save utilization statistics after deallocation
    topology.save_stats_to_csv('topology_utilization_after_deallocation.csv')

    # Verify that the network is back to its original state
    network_state_ok = topology.verify_network_state()
    # print(f"Network state is back to original: {network_state_ok}")
