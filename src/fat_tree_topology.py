import networkx as nx
from topology_nx import BaseTopology  

class FatTreeTopology(BaseTopology):
    def __init__(self, num_pods=4, bandwidth=10):
        super().__init__()
        self.num_pods = num_pods
        self.bandwidth = bandwidth
        self.create_topology()

    def create_topology(self):
        """
        Build the Fat-Tree topology with core, aggregation, and edge layers.
        """
        num_core_switches = (self.num_pods // 2) ** 2
        num_agg_switches = self.num_pods * (self.num_pods // 2)
        num_edge_switches = num_agg_switches
        num_hosts = num_edge_switches * (self.num_pods // 2)

        core_switches = [f"core_{i}" for i in range(num_core_switches)]
        for switch in core_switches:
            self.G.add_node(switch, type="core")

        agg_switches = [f"agg_{i}" for i in range(num_agg_switches)]
        for i, agg in enumerate(agg_switches):
            self.G.add_node(agg, type="aggregation")

            
            for j in range(self.num_pods // 2):
                core_switch = core_switches[(i % (num_core_switches // (self.num_pods // 2))) + j]
                self.G.add_edge(agg, core_switch, bandwidth=self.bandwidth)

        edge_switches = [f"edge_{i}" for i in range(num_edge_switches)]
        for i, edge in enumerate(edge_switches):
            self.G.add_node(edge, type="edge")
            for j in range(self.num_pods // 2):
                agg = agg_switches[(i // (self.num_pods // 2)) * (self.num_pods // 2) + j]
                self.G.add_edge(edge, agg, bandwidth=self.bandwidth)

        host_id = 0
        for edge in edge_switches:
            for _ in range(self.num_pods // 2):
                host = f"host_{host_id}"
                self.G.add_node(host, type="host")
                self.G.add_edge(host, edge, bandwidth=self.bandwidth)
                host_id += 1

    def allocate_ps_to_workers_balanced(self):
        """
        Allocate processing resources (ps) to workers in a balanced way.
        This method should consider the FatTree topology and the bandwidth constraints.
        """
        
        core_switches = [node for node, data in self.G.nodes(data=True) if data['type'] == 'core']
        agg_switches = [node for node, data in self.G.nodes(data=True) if data['type'] == 'aggregation']
        edge_switches = [node for node, data in self.G.nodes(data=True) if data['type'] == 'edge']
        hosts = [node for node, data in self.G.nodes(data=True) if data['type'] == 'host']

        
        total_bandwidth = sum([self.G[host][edge].get('bandwidth', self.bandwidth) for host in hosts for edge in edge_switches])
        avg_bandwidth = total_bandwidth / len(hosts)

        allocation = {}
        for host in hosts:
            
            allocated_switch = self._find_best_edge_switch(host, avg_bandwidth)
            allocation[host] = allocated_switch
            self.G.nodes[allocated_switch]['allocated'] = self.G.nodes.get(allocated_switch, {}).get('allocated', 0) + avg_bandwidth
        return allocation

    def _find_best_edge_switch(self, host, avg_bandwidth):
        """
        Helper function to find the best edge switch based on available bandwidth.
        This function ensures the resources are allocated in a way that maximizes bandwidth efficiency.
        """
        min_bandwidth = float('inf')
        best_switch = None
        for edge in self.G.neighbors(host):
            if self.G[host][edge].get('bandwidth', self.bandwidth) < min_bandwidth:
                min_bandwidth = self.G[host][edge].get('bandwidth', self.bandwidth)
                best_switch = edge
        return best_switch

    def allocate_ps_to_workers_single(self):
        """
        Allocate processing resources (ps) to workers in a single allocation.
        This method should allocate resources to the workers in a way that respects the FatTree topology.
        """
        
        allocation = {}
        for host in self.G.nodes(data=True):
            if host[1]['type'] == 'host':
                allocated_switch = self._find_best_edge_switch(host[0], self.bandwidth)
                allocation[host[0]] = allocated_switch
        return allocation

if __name__ == "__main__":
    fat_tree = FatTreeTopology(num_pods=4, bandwidth=10)
    
    print("Nodes in the Fat-Tree Topology:")
    print(fat_tree.G.nodes(data=True))
    print("\nEdges in the Fat-Tree Topology:")
    print(fat_tree.G.edges(data=True))

    
    balanced_allocation = fat_tree.allocate_ps_to_workers_balanced()
    print("\nBalanced Allocation:")
    print(balanced_allocation)

    single_allocation = fat_tree.allocate_ps_to_workers_single()
    print("\nSingle Allocation:")
    print(single_allocation)
