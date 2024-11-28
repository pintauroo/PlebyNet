from network_topology import FatTreeTopology
from visualization import plot_topology
k = 4
num_core = 4
num_agg = 8
num_edge = 8
num_hosts = 16
fat_tree = FatTreeTopology(num_core, num_agg, num_edge, num_hosts)
graph = fat_tree.create_topology()
plot_topology(graph, title="Fat-Tree Topology Example", save_path="fat_tree_topology.png")
