import networkx as nx
import numpy as np
import pandas as pd
import logging

# Suppress matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class BaseTopology:
    def __init__(self):
        self.G = nx.Graph()
        self.allocated_paths = {}  # key: job_id, value: dict of allocations
        self.utilization_records = []
        self.allocation_step = 0
        self.adj = None  # Initialize adjacency matrix

    def create_topology(self):
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
                    path = nx.shortest_path(self.G, source=src, target=dst)
                    min_available_bw = float('inf')

                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]

                        # Edge data
                        edge_data = self.G.get_edge_data(u, v)
                        bandwidth = edge_data.get('bandwidth', 0.0)
                        reserved_bw = edge_data.get('reserved_bw', 0.0)
                        available_bw_edge = max(float(bandwidth) - float(reserved_bw), 0.0)

                        # Node data for u
                        if self.G.nodes[u]['type'] != 'host':
                            node_bandwidth = self.G.nodes[u].get('bandwidth', float('inf'))
                            node_reserved_bw = self.G.nodes[u].get('reserved_bw', 0.0)
                            available_bw_node_u = max(float(node_bandwidth) - float(node_reserved_bw), 0.0)
                        else:
                            available_bw_node_u = float('inf')

                        # Node data for v
                        if self.G.nodes[v]['type'] != 'host':
                            node_bandwidth = self.G.nodes[v].get('bandwidth', float('inf'))
                            node_reserved_bw = self.G.nodes[v].get('reserved_bw', 0.0)
                            available_bw_node_v = max(float(node_bandwidth) - float(node_reserved_bw), 0.0)
                        else:
                            available_bw_node_v = float('inf')

                        # The available bandwidth on this segment
                        available_bw = min(available_bw_edge, available_bw_node_u, available_bw_node_v)
                        min_available_bw = min(min_available_bw, available_bw)

                    adj_matrix[i][j] = min_available_bw

                except nx.NetworkXNoPath:
                    adj_matrix[i][j] = 0.0

        return adj_matrix

    def record_utilization(self, time_instant, job_id):
        record = {'allocation_step': time_instant, 'job_id': job_id}

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

    def allocate_bandwidth_between_workers_and_ps(self, allocation_list, total_required_bw, job_id, time_instant, allow_oversubscription=False):
        if not allocation_list:
            return 0.0

        ps_nodes = set(allocation_list)
        num_ps_nodes = len(ps_nodes)
        required_per_connection = float(total_required_bw) / num_ps_nodes  # Distribute total BW equally among PS nodes

        # Prepare to allocate bandwidth
        modifications = []
        paths = {}
        edge_usage_count = {}
        node_usage_count = {}

        # For each worker-PS pair, allocate bandwidth
        for worker_id in allocation_list:
            worker = self.format_node_id(worker_id)
            for ps_id in ps_nodes:
                ps = self.format_node_id(ps_id)

                if worker == ps:
                    # No network bandwidth is consumed if worker and PS are on the same host
                    continue

                try:
                    path = nx.shortest_path(self.G, source=worker, target=ps)
                    key = (worker, ps)
                    paths[key] = path

                    # Update node usage counts
                    for node in path:
                        if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                            node_usage_count[node] = node_usage_count.get(node, 0) + 1

                    # Update edge usage counts
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge = (u, v) if self.G.has_edge(u, v) else (v, u)
                        edge_usage_count[edge] = edge_usage_count.get(edge, 0) + 1

                except nx.NetworkXNoPath:
                    print(f"No path found between worker {worker} and PS {ps}.")
                    return 0.0

        # Compute the maximum per-connection bandwidth
        max_bw_per_connection = float('inf')

        for node, usage in node_usage_count.items():
            available_bw = float(self.G.nodes[node]['bandwidth']) - float(self.G.nodes[node]['reserved_bw'])
            per_connection_bw = available_bw / usage
            if per_connection_bw < max_bw_per_connection:
                max_bw_per_connection = per_connection_bw

        for edge, usage in edge_usage_count.items():
            u, v = edge
            available_bw = float(self.G[u][v]['bandwidth']) - float(self.G[u][v]['reserved_bw'])
            per_connection_bw = available_bw / usage
            if per_connection_bw < max_bw_per_connection:
                max_bw_per_connection = per_connection_bw

        # Limit allocated bandwidth per connection to the required per connection
        allocated_per_connection_bw = min(max_bw_per_connection, required_per_connection)

        if allocated_per_connection_bw <= 0.0:
            # print("No bandwidth available for allocation.")
            return 0.0

        # Proceed to allocate bandwidth
        try:
            for (worker, ps), path in paths.items():
                total_bw = allocated_per_connection_bw

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        new_reserved_bw = float(self.G.nodes[node]['reserved_bw']) + total_bw
                        if new_reserved_bw > float(self.G.nodes[node]['bandwidth']):
                            if not allow_oversubscription:
                                raise Exception(f"Not enough bandwidth on node {node} during allocation.")
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] = round(new_reserved_bw, 6)
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    new_reserved_bw = float(self.G[u][v]['reserved_bw']) + total_bw
                    if new_reserved_bw > float(self.G[u][v]['bandwidth']):
                        if not allow_oversubscription:
                            raise Exception(f"Not enough bandwidth on link {u}-{v} during allocation.")
                    old_reserved_bw = self.G[u][v]['reserved_bw']
                    self.G[u][v]['reserved_bw'] = round(new_reserved_bw, 6)
                    modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))

            # Record the allocation under the job_id
            if job_id not in self.allocated_paths:
                self.allocated_paths[job_id] = {}

            for (worker, ps), path in paths.items():
                self.allocated_paths[job_id][(worker, ps)] = (path, allocated_per_connection_bw)

            self.record_utilization(time_instant, job_id)
            # Recalculate the adjacency matrix
            self.adj = self.calculate_host_to_host_adjacency_matrix()
            return allocated_per_connection_bw

        except Exception as e:
            # print(f"Exception occurred during allocation: {e}")
            # Rollback any changes due to exception
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return 0.0

    def deallocate_bandwidth_between_workers_and_ps(self, job_id, time_instant):
        if job_id not in self.allocated_paths:
            # print(f"No allocations found for job_id {job_id}")
            return False

        allocations = self.allocated_paths[job_id]
        modifications = []
        try:
            for (worker, ps), (path, allocated_bw) in allocations.items():
                total_bw = allocated_bw

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        new_reserved_bw = float(self.G.nodes[node]['reserved_bw']) - total_bw
                        self.G.nodes[node]['reserved_bw'] = max(round(new_reserved_bw, 6), 0.0)
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    old_reserved_bw = self.G[u][v]['reserved_bw']
                    new_reserved_bw = float(self.G[u][v]['reserved_bw']) - total_bw
                    self.G[u][v]['reserved_bw'] = max(round(new_reserved_bw, 6), 0.0)
                    modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))

            # After successful deallocation, remove the job_id from allocated_paths
            del self.allocated_paths[job_id]
            self.record_utilization(time_instant, job_id)
            # Recalculate the adjacency matrix
            self.adj = self.calculate_host_to_host_adjacency_matrix()
            return True
        except Exception as e:
            print(f"Exception occurred during deallocation: {e}")
            # In case of exception, roll back changes
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return False

    def format_node_id(self, node_id):
        return f"H{str(node_id)}"

    def format_node_ids(self, node_ids):
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        return [f"H{str(node_id)}" for node_id in node_ids]

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
            G.add_node(spine, type='spine', bandwidth=float(spine_bw), reserved_bw=0.0)

        leaf_switches = [f"L{i}" for i in range(num_leaf)]
        for leaf in leaf_switches:
            G.add_node(leaf, type='leaf', bandwidth=float(leaf_bw), reserved_bw=0.0)

        for leaf in leaf_switches:
            for spine in spine_switches:
                G.add_edge(leaf, spine, bandwidth=float(link_bw_leaf_to_spine), reserved_bw=0.0)

        host_id = 0
        for leaf in leaf_switches:
            for _ in range(num_hosts_per_leaf):
                host = f"H{host_id}"
                G.add_node(host, type='host', bandwidth=float(link_bw_leaf_to_node), reserved_bw=0.0)
                G.add_edge(leaf, host, bandwidth=float(link_bw_leaf_to_node), reserved_bw=0.0)
                host_id += 1

        return G

    def verify_network_state(self):
        epsilon = 1e-6  # Tolerance level for floating-point comparison
        # Check nodes
        for node, data in self.G.nodes(data=True):
            original_bw = self.original_reserved_bw_nodes.get(node, 0.0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                print(f"Node {node} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # Check edges
        for u, v, data in self.G.edges(data=True):
            edge = (u, v)
            original_bw = self.original_reserved_bw_edges.get(edge, 0.0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                print(f"Edge {u}-{v} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # print("Network state verification passed. The network is back to its original state.")
        return True

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

    # First test case
    allocation_list = [i for i in range(1, 11)]  # Workers 0-9 assigned to PS nodes 1-10
    total_bw = 100  # Total bandwidth required
    job_id = 1

    # Define time instants
    time_instant_allocation = 0
    time_instant_deallocation = 1

    print("Adjacency Matrix Before Allocation:")
    print(topology.adj)

    # Allocate resources for the job
    allocated_per_conn = topology.allocate_bandwidth_between_workers_and_ps(
        allocation_list, total_bw, job_id, time_instant=time_instant_allocation, allow_oversubscription=False
    )
    print(f"Allocation success for job {job_id}: {allocated_per_conn > 0}")
    print(f"Allocated bandwidth per connection for job {job_id}: {allocated_per_conn}")

    print("Adjacency Matrix After Allocation:")
    print(topology.adj)

    # Deallocate resources for the job
    deallocation_success = topology.deallocate_bandwidth_between_workers_and_ps(job_id, time_instant_deallocation)
    print(f"Deallocation success for job {job_id}: {deallocation_success}")

    # Verify that the network is back to its original state
    network_state_ok = topology.verify_network_state()
    print(f"Network state is back to original: {network_state_ok}")
