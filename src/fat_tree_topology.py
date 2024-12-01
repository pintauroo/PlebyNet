import argparse
import networkx as nx
import matplotlib.pyplot as plt
from topology_nx import BaseTopology


class FatTreeTopology(BaseTopology):
    def __init__(self, num_pods=4, bandwidth=10, custom_bandwidth=None, custom_connections=None):
        """
        Initialize the FatTreeTopology class.

        :param num_pods: Number of pods in the Fat-Tree topology
        :param bandwidth: Default bandwidth for connections
        :param custom_bandwidth: Dictionary to define bandwidth between specific node types
        :param custom_connections: Dictionary defining custom connection rules between node types
        """
        super().__init__()
        self.num_pods = num_pods
        self.bandwidth = bandwidth
        self.custom_bandwidth = custom_bandwidth or {}
        self.custom_connections = custom_connections or {}
        self.create_topology()

    def create_topology(self):
        """
        Build the Fat-Tree topology with generalized node types and connections.
        """
        num_core_switches = (self.num_pods // 2) ** 2
        num_agg_switches = self.num_pods * (self.num_pods // 2)
        num_edge_switches = num_agg_switches
        num_hosts = num_edge_switches * (self.num_pods // 2)

        # Add core switches
        core_switches = [f"core_{i}" for i in range(num_core_switches)]
        self._add_nodes(core_switches, node_type="core")

        # Add aggregation switches
        agg_switches = [f"agg_{i}" for i in range(num_agg_switches)]
        self._add_nodes(agg_switches, node_type="aggregation")

        # Connect core switches to aggregation switches
        for i, agg in enumerate(agg_switches):
            for j in range(self.num_pods // 2):
                core = core_switches[(i % (num_core_switches // (self.num_pods // 2))) + j]
                self._add_edge(agg, core, node_types=("aggregation", "core"))

        # Add edge switches
        edge_switches = [f"edge_{i}" for i in range(num_edge_switches)]
        self._add_nodes(edge_switches, node_type="edge")

        # Connect aggregation switches to edge switches
        for i, edge in enumerate(edge_switches):
            for j in range(self.num_pods // 2):
                agg = agg_switches[(i // (self.num_pods // 2)) * (self.num_pods // 2) + j]
                self._add_edge(edge, agg, node_types=("edge", "aggregation"))

        # Add hosts
        host_id = 0
        for edge in edge_switches:
            for _ in range(self.num_pods // 2):
                host = f"host_{host_id}"
                self._add_nodes([host], node_type="host")
                self._add_edge(host, edge, node_types=("host", "edge"))
                host_id += 1

    def _add_nodes(self, nodes, node_type):
        """
        Add nodes to the graph with a specific type.
        """
        for node in nodes:
            self.G.add_node(node, type=node_type)

    def _add_edge(self, node1, node2, node_types):
        """
        Add edges between nodes, considering custom bandwidth.
        """
        default_bandwidth = self.custom_bandwidth.get(node_types, self.bandwidth)
        self.G.add_edge(node1, node2, bandwidth=default_bandwidth, available_bandwidth=default_bandwidth, used_bandwidth=0)

    def allocate_ps_to_workers_balanced(self):
        """
        Allocate processing resources (ps) to workers in a balanced way.
        """
        allocation = {}
        hosts = [node for node, data in self.G.nodes(data=True) if data['type'] == 'host']
        for host in hosts:
            best_switch = self._find_best_edge_switch(host)
            if best_switch:
                allocation[host] = best_switch
                if self.G[host][best_switch]['available_bandwidth'] >= self.bandwidth:
                    self.G[host][best_switch]['available_bandwidth'] -= self.bandwidth
                    self.G[host][best_switch]['used_bandwidth'] += self.bandwidth
                    print(f"Allocated {self.bandwidth} bandwidth from {best_switch} to {host}")
                    print(f"Updated available bandwidth: {self.G[host][best_switch]['available_bandwidth']}")
                else:
                    print(f"Not enough bandwidth to allocate from {best_switch} to {host}")
        return allocation

    def allocate_ps_to_workers_single(self, node, workers):
        """
        Allocate processing resources (ps) to specific workers.
        """
        allocation = {}
        for worker in workers:
            if self.G.has_edge(node, worker) and self.G[node][worker]['available_bandwidth'] > 0:
                allocation[worker] = node
                self.G[node][worker]['available_bandwidth'] -= self.bandwidth
                self.G[node][worker]['used_bandwidth'] += self.bandwidth
            else:
                allocation[worker] = None
        return allocation

    def deallocate_bandwidth(self, node, workers):
        """
        Deallocate resources by resetting bandwidth.
        """
        for worker in workers:
            if self.G.has_edge(node, worker):
                self.G[node][worker]['available_bandwidth'] += self.bandwidth
                self.G[node][worker]['used_bandwidth'] -= self.bandwidth

    def _find_best_edge_switch(self, host):
        """
        Find the best edge switch for a given host based on available bandwidth.
        """
        best_switch = None
        max_bandwidth = -1
        for neighbor in self.G.neighbors(host):
            edge_data = self.G[host][neighbor]
            if edge_data.get('available_bandwidth', 0) > max_bandwidth:
                max_bandwidth = edge_data['available_bandwidth']
                best_switch = neighbor
        return best_switch

    def get_utilization_stats(self):
        """
        Calculate and return the bandwidth utilization statistics for the entire topology.
        """
        stats = {
            'total_bandwidth': 0,
            'used_bandwidth': 0,
            'available_bandwidth': 0,
            'utilization_percentage': 0
        }
        
        # Loop over each edge and calculate the stats
        for node1, node2, data in self.G.edges(data=True):
            total_bandwidth = data['bandwidth']
            available_bandwidth = data['available_bandwidth']
            used_bandwidth = data.get('used_bandwidth', 0)  # Track the used bandwidth correctly
            
            stats['total_bandwidth'] += total_bandwidth
            stats['used_bandwidth'] += used_bandwidth
            stats['available_bandwidth'] += available_bandwidth
    
        # Calculate utilization as a percentage
        if stats['total_bandwidth'] > 0:
            stats['utilization_percentage'] = (stats['used_bandwidth'] / stats['total_bandwidth']) * 100

        return stats

    def visualize_topology(self):
        """
        Visualize the Fat-Tree topology with color coding for different node types.
        """
        # Use a shell layout to position nodes
        shell_pos = nx.shell_layout(self.G)  
        
        # Draw the nodes and edges
        node_colors = [self._get_node_color(node) for node in self.G.nodes()]
        nx.draw(self.G, pos=shell_pos, with_labels=True, node_size=500, node_color=node_colors, font_size=8, font_weight="bold", edge_color='gray')

        # Show the plot
        plt.title("Fat-Tree Topology Visualization")
        plt.show()

    def _get_node_color(self, node):
        """
        Return the color for each node type.
        """
        node_type = self.G.nodes[node]['type']
        if node_type == "core":
            return 'red'
        elif node_type == "aggregation":
            return 'blue'
        elif node_type == "edge":
            return 'green'
        else:
            return 'yellow'


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Configure Fat-Tree Topology")
    parser.add_argument("--num_pods", type=int, default=4, help="Number of pods in the Fat-Tree topology")
    parser.add_argument("--bandwidth", type=int, default=10, help="Default bandwidth for connections")
    args = parser.parse_args()

    # Initialize topology
    fat_tree = FatTreeTopology(num_pods=args.num_pods, bandwidth=args.bandwidth)

    # Display nodes and edges
    print("Nodes in the Fat-Tree Topology:")
    print(fat_tree.G.nodes(data=True))
    print("\nEdges in the Fat-Tree Topology:")
    print(fat_tree.G.edges(data=True))

    # Perform balanced allocation
    balanced_allocation = fat_tree.allocate_ps_to_workers_balanced()
    print("\nBalanced Allocation:")
    print(balanced_allocation)

    # Reset bandwidths for testing
    for node1, node2, data in fat_tree.G.edges(data=True):
        data['available_bandwidth'] = data['bandwidth']

    # Perform single allocation
    node = "edge_0"
    workers = ["host_0", "host_1"]
    single_allocation = fat_tree.allocate_ps_to_workers_single(node, workers)
    print("\nSingle Allocation:")
    print(single_allocation)

    # Deallocate resources
    fat_tree.deallocate_bandwidth(node, workers)
    print("\nDeallocated Bandwidth for workers from edge_0:")
    print(f"Updated Bandwidths: {[fat_tree.G[node][worker]['available_bandwidth'] for worker in workers if fat_tree.G.has_edge(node, worker)]}")

    # Get utilization stats
    utilization_stats = fat_tree.get_utilization_stats()
    
    # Display stats
    print("\nBandwidth Utilization Stats:")
    print(f"Total Bandwidth: {utilization_stats['total_bandwidth']}")
    print(f"Used Bandwidth: {utilization_stats['used_bandwidth']}")
    print(f"Available Bandwidth: {utilization_stats['available_bandwidth']}")
    print(f"Utilization: {utilization_stats['utilization_percentage']}%")

    # Visualize the topology
    fat_tree.visualize_topology()
