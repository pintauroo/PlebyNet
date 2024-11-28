import unittest
import networkx as nx  
from src.network_topology import FatTreeTopology
from src.visualization import plot_topology
class TestFatTreeTopology(unittest.TestCase):
    
    def setUp(self):
        self.k = 4  
        self.num_core = 4
        self.num_agg = 8
        self.num_edge = 8
        self.num_hosts = 16
        self.fat_tree = FatTreeTopology(self.num_core, self.num_agg, self.num_edge, self.num_hosts)
        self.graph = self.fat_tree.create_topology()

    def test_create_topology_structure(self):
        expected_num_nodes = self.num_core + self.num_agg + self.num_edge + self.num_hosts
        self.assertEqual(len(self.graph.nodes), expected_num_nodes)
        
        core_agg_edges = [(u, v) for u, v in self.graph.edges if 'core' in u and 'agg' in v]
        agg_edge_edges = [(u, v) for u, v in self.graph.edges if 'agg' in u and 'edge' in v]
        edge_host_edges = [(u, v) for u, v in self.graph.edges if 'edge' in u and 'host' in v]
        
        self.assertTrue(len(core_agg_edges) > 0, "No core-to-aggregation layer edges found")
        self.assertTrue(len(agg_edge_edges) > 0, "No aggregation-to-edge layer edges found")
        self.assertTrue(len(edge_host_edges) > 0, "No edge-to-host layer edges found")
    
    def test_node_types(self):
        core_nodes = [n for n in self.graph.nodes if 'core' in n]
        agg_nodes = [n for n in self.graph.nodes if 'agg' in n]
        edge_nodes = [n for n in self.graph.nodes if 'edge' in n]
        host_nodes = [n for n in self.graph.nodes if 'host' in n]
        
        self.assertEqual(len(core_nodes), self.num_core, "Mismatch in core node count")
        self.assertEqual(len(agg_nodes), self.num_agg, "Mismatch in aggregation node count")
        self.assertEqual(len(edge_nodes), self.num_edge, "Mismatch in edge node count")
        self.assertEqual(len(host_nodes), self.num_hosts, "Mismatch in host node count")

    def test_path_availability(self):
        for host in [n for n in self.graph.nodes if 'host' in n]:
            reachable_from_host = False
            for core in [n for n in self.graph.nodes if 'core' in n]:
                if nx.has_path(self.graph, host, core):  
                    reachable_from_host = True
                    break
            self.assertTrue(reachable_from_host, f"No path found from host {host} to any core node")
    
    def test_edge_capacities(self):
        for u, v, data in self.graph.edges(data=True):
            self.assertIn('bandwidth', data, f"Edge ({u}, {v}) is missing bandwidth attribute")
            self.assertGreater(data['bandwidth'], 0, f"Edge ({u}, {v}) has non-positive bandwidth")

    def test_visualization_support(self):
        try:
            plot_topology(self.graph)
        except ImportError:
            self.fail("Visualization module not found")
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")
    
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
