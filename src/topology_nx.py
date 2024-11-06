import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BaseTopology:
    def __init__(self):
        self.G = nx.Graph()
        self.allocated_paths = {}
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
        print(f"Utilization statistics saved to {filename}")

    def allocate_ps_to_workers_balanced(self, worker_nodes, required_bw, allow_oversubscription=False):
        worker_nodes_formatted = self.format_node_ids(worker_nodes)
        if not worker_nodes_formatted:
            print("No worker nodes provided for allocation.")
            return False
        ps_node = worker_nodes_formatted[0]
        actual_worker_nodes = worker_nodes_formatted[1:]

        if not actual_worker_nodes:
            print("No worker nodes provided after specifying PS node.")
            return False

        ps_nodes_formatted = set([ps_node])
        paths = {}
        ps_worker_pairs = set()

        for ps in ps_nodes_formatted:
            for worker in actual_worker_nodes:
                if worker != ps:
                    ps_worker_pairs.add((ps, worker))

        tentative_reserved_bw_nodes = {}
        tentative_reserved_bw_edges = {}

        for u, v, data in self.G.edges(data=True):
            existing_reserved_bw = data['reserved_bw']
            available_bw = data['bandwidth'] - existing_reserved_bw
            if available_bw > 0:
                data['weight'] = 1 / available_bw
            else:
                data['weight'] = float('inf')

        for (ps_node, worker_node) in ps_worker_pairs:
            try:
                path = nx.shortest_path(self.G, source=ps_node, target=worker_node, weight='weight')
                paths[(ps_node, worker_node)] = path

                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        existing_reserved_bw = self.G.nodes[node]['reserved_bw']
                        tentative_bw = tentative_reserved_bw_nodes.get(node, 0)
                        total_reserved_bw = existing_reserved_bw + tentative_bw + required_bw
                        if total_reserved_bw > self.G.nodes[node]['bandwidth']:
                            print(
                                f"Insufficient bandwidth on node {node}. "
                                f"Cannot allocate {required_bw} Gbps."
                            )
                            return False
                        tentative_reserved_bw_nodes[node] = tentative_bw + required_bw

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = frozenset([u, v])
                    existing_reserved_bw = self.G[u][v]['reserved_bw']
                    tentative_bw = tentative_reserved_bw_edges.get(edge, 0)
                    total_reserved_bw = existing_reserved_bw + tentative_bw + required_bw
                    if total_reserved_bw > self.G[u][v]['bandwidth']:
                        print(
                            f"Insufficient bandwidth on link {u}-{v}. "
                            f"Cannot allocate {required_bw} Gbps."
                        )
                        return False
                    tentative_reserved_bw_edges[edge] = tentative_bw + required_bw

            except nx.NetworkXNoPath:
                print(f"No path found between PS {ps_node} and worker {worker_node}.")
                return False

        for node, bw in tentative_reserved_bw_nodes.items():
            self.G.nodes[node]['reserved_bw'] += bw
            assert self.G.nodes[node]['reserved_bw'] <= self.G.nodes[node]['bandwidth'], (
                f"Node {node} over-allocated after reservation. "
                f"Reserved: {self.G.nodes[node]['reserved_bw']} Gbps, "
                f"Capacity: {self.G.nodes[node]['bandwidth']} Gbps"
            )

        for edge, bw in tentative_reserved_bw_edges.items():
            u, v = tuple(edge)
            self.G[u][v]['reserved_bw'] += bw
            assert self.G[u][v]['reserved_bw'] <= self.G[u][v]['bandwidth'], (
                f"Edge {u}-{v} over-allocated after reservation. "
                f"Reserved: {self.G[u][v]['reserved_bw']} Gbps, "
                f"Capacity: {self.G[u][v]['bandwidth']} Gbps"
            )

        self.allocated_paths.update(paths)
        self.record_utilization()

        return True

    def allocate_ps_to_workers_single(self, ps_node, worker_nodes, required_bw, allow_oversubscription=False):
        ps_node = self.format_node_ids([ps_node])[0]
        worker_nodes = self.format_node_ids(worker_nodes)
        paths = {}
        cumulative_node_bw = {}
        cumulative_edge_bw = {}
        allocated_nodes = set()
        allocated_edges = []
        modifications = []

        for worker in worker_nodes:
            if ps_node == worker:
                continue

            try:
                path = nx.shortest_path(self.G, source=ps_node, target=worker)
                paths[worker] = path

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

        try:
            for worker in worker_nodes:
                if ps_node == worker:
                    continue

                for node in paths[worker]:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        new_reserved_bw = self.G.nodes[node]['reserved_bw'] + required_bw
                        if new_reserved_bw > self.G.nodes[node]['bandwidth']:
                            raise Exception(f"Not enough bandwidth on node {node} during allocation.")
                        old_reserved_bw = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] = new_reserved_bw
                        modifications.append(('node', node, 'reserved_bw', old_reserved_bw))
                        allocated_nodes.add(node)

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

            for worker_node in worker_nodes:
                if ps_node == worker_node:
                    continue
                self.allocated_paths[(ps_node, worker_node)] = paths[worker_node]

            self.record_utilization()

            return True
        except Exception as e:
            print(f"Exception occurred during allocation: {e}")
            for obj_type, obj_id, attr, old_value in reversed(modifications):
                if obj_type == 'node':
                    self.G.nodes[obj_id][attr] = old_value
                elif obj_type == 'edge':
                    u, v = obj_id
                    self.G[u][v][attr] = old_value
            return False

    def deallocate_ps_from_workers(self, worker_nodes, required_bw):
        worker_nodes_formatted = self.format_node_ids(worker_nodes)
        if not worker_nodes_formatted:
            print("No worker nodes provided for deallocation.")
            return False
        ps_node = worker_nodes_formatted[0]
        actual_worker_nodes = worker_nodes_formatted[1:]

        if not actual_worker_nodes:
            print("No worker nodes provided after specifying PS node.")
            return False

        ps_nodes_formatted = set([ps_node])
        rollback_required = False
        initial_node_reserved_bw = {}
        initial_edge_reserved_bw = {}
        paths_to_remove = []

        for (ps_node_key, worker_node), path in list(self.allocated_paths.items()):
            if ps_node_key in ps_nodes_formatted and worker_node in actual_worker_nodes:
                for node in path:
                    if self.G.nodes[node]['type'] in ['spine', 'leaf']:
                        initial_node_reserved_bw[node] = self.G.nodes[node]['reserved_bw']
                        self.G.nodes[node]['reserved_bw'] -= required_bw
                        self.G.nodes[node]['reserved_bw'] = max(self.G.nodes[node]['reserved_bw'], 0)

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = (u, v) if (u, v) in self.G.edges else (v, u)
                    initial_edge_reserved_bw[edge] = self.G[u][v]['reserved_bw']
                    self.G[u][v]['reserved_bw'] -= required_bw
                    self.G[u][v]['reserved_bw'] = max(self.G[u][v]['reserved_bw'], 0)

                paths_to_remove.append((ps_node_key, worker_node))

        for key in paths_to_remove:
            del self.allocated_paths[key]

        return True

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

    adj_matrix = topology.calculate_host_to_host_adjacency_matrix()
    print("Adjacency Matrix:", adj_matrix)

    topology.save_stats_to_csv('topology_utilization_stats.csv')
