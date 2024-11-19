import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BaseTopology:
    def __init__(self):
        self.G = nx.Graph()
        self.allocated_paths = {}  # Maps job_id to list of (ps_node, worker_node, path, allocated_bw)
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

    def record_utilization(self, time_instant, job_id):
        record = {'allocation_step': time_instant,
                  'job_id': job_id}
        
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

    def deallocate_ps_from_workers(self, job_id, time_instant):
        allocations = self.allocated_paths.get(job_id, [])
        if not allocations:
            # print(f"No allocations found for job_id {job_id}")
            return False

        for ps_node, worker_node, path, allocated_bw in allocations:
            # Deallocate resources along the path
            for node in path:
                if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                    self.G.nodes[node]['reserved_bw'] -= allocated_bw
                    self.G.nodes[node]['reserved_bw'] = max(self.G.nodes[node]['reserved_bw'], 0)

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = (u, v) if (u, v) in self.G.edges else (v, u)
                self.G[u][v]['reserved_bw'] -= allocated_bw
                self.G[u][v]['reserved_bw'] = max(self.G[u][v]['reserved_bw'], 0)

        # Remove allocations for this job_id
        del self.allocated_paths[job_id]
        self.record_utilization(time_instant,job_id)
        return True

    def format_node_ids(self, node_ids):
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        return [f"H{str(node_id)}" if not str(node_id).startswith('H') else str(node_id) for node_id in node_ids]

    def compute_max_allocatable_bw(self, ps_node, worker_nodes, required_bw, job_id, time_instant, max_bw=None, allow_oversubscription=False):
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
                print(f"No path found between PS {ps_node} and worker {worker}.")
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

        if max_bw_per_worker >= required_bw / 3:
            # Proceed to allocate max_bw_per_worker to each path
            try:
                for worker in worker_nodes:
                    if ps_node == worker:
                        continue

                    path = paths[worker]

                    for node in path:
                        if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                            new_reserved_bw = self.G.nodes[node]['reserved_bw'] + max_bw_per_worker
                            if new_reserved_bw > self.G.nodes[node]['bandwidth']:
                                raise Exception(f"Not enough bandwidth on node {node} during allocation.")
                            old_reserved_bw = self.G.nodes[node]['reserved_bw']
                            self.G.nodes[node]['reserved_bw'] = new_reserved_bw
                            modifications.append(('node', node, 'reserved_bw', old_reserved_bw))

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        new_reserved_bw = self.G[u][v]['reserved_bw'] + max_bw_per_worker
                        if new_reserved_bw > self.G[u][v]['bandwidth']:
                            if not allow_oversubscription:
                                raise Exception(f"Not enough bandwidth on link {u}-{v} during allocation.")
                        old_reserved_bw = self.G[u][v]['reserved_bw']
                        self.G[u][v]['reserved_bw'] = new_reserved_bw
                        modifications.append(('edge', (u, v), 'reserved_bw', old_reserved_bw))

                # Store allocations under the job_id, including allocated_bw
                if job_id not in self.allocated_paths:
                    self.allocated_paths[job_id] = []
                for worker in worker_nodes:
                    if ps_node == worker:
                        continue
                    path = paths[worker]
                    self.allocated_paths[job_id].append((ps_node, worker, path, max_bw_per_worker))

                self.record_utilization(time_instant,job_id)

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
                print(f"Node {node} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # Check edges
        for u, v, data in self.G.edges(data=True):
            edge = (u, v)
            original_bw = self.original_reserved_bw_edges.get(edge, 0)
            current_bw = data['reserved_bw']
            if abs(original_bw - current_bw) > epsilon:
                print(f"Edge {u}-{v} reserved_bw mismatch. Original: {original_bw}, Current: {current_bw}")
                return False

        # print("Network state verification passed. The network is back to its original state.")
        return True

# Example usage
if __name__ == "__main__":
    # Parameters for Medium Deployment
    num_spine_switches = 4
    num_leaf_switches = 6
    num_hosts_per_leaf = 8
    
    spine_bandwidth = 200
    leaf_bandwidth = 400
    link_bw_leaf_to_node = 100
    link_bw_leaf_to_spine = 100

    topology = SpineLeafTopology(
        num_spine_switches, num_leaf_switches, num_hosts_per_leaf,
        spine_bandwidth, leaf_bandwidth, link_bw_leaf_to_node, link_bw_leaf_to_spine
    )

    # Adjacency matrix
    adj_matrix = topology.calculate_host_to_host_adjacency_matrix()
    print("Adjacency Matrix:\n", adj_matrix)

    # Allocate resources for job_id 1
    job_id = 1
    ps_node = 'H0'
    worker_nodes = ['H1', 'H2', 'H3']
    required_bw = 50  # Bandwidth required per worker

    allocated_bw = topology.compute_max_allocatable_bw(ps_node, worker_nodes, required_bw, job_id)
    if allocated_bw >= required_bw / 3:
        print(f"Allocated bandwidth per worker for job {job_id}: {allocated_bw}")
    else:
        print(f"Failed to allocate required bandwidth for job {job_id}")

    # Deallocate resources for job_id 1
    deallocated = topology.deallocate_ps_from_workers(job_id)
    if deallocated:
        print(f"Successfully deallocated resources for job {job_id}")
    else:
        print(f"Failed to deallocate resources for job {job_id}")

    topology.verify_network_state()

    # Save utilization stats
    topology.save_stats_to_csv('topology_utilization_stats.csv')
