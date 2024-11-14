# Creating the `test_topology.py` file with tests for the FatTreeTopology implementation.
import sys
import os
import unittest

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from topology_nx import FatTreeTopology  # Import the required class


class TestFatTreeTopology(unittest.TestCase):

    def setUp(self):
        # Initialize the FatTreeTopology with test parameters
        self.num_core_switches = 2
        self.num_aggregation_switches = 4
        self.num_edge_switches = 8
        self.hosts_per_edge_switch = 10
        self.core_bw = 200
        self.agg_bw = 100
        self.edge_bw = 50

        self.topology = FatTreeTopology(
            num_core=self.num_core_switches,
            num_aggregation=self.num_aggregation_switches,
            num_edge=self.num_edge_switches,
            hosts_per_edge=self.hosts_per_edge_switch,
            core_bw=self.core_bw,
            agg_bw=self.agg_bw,
            edge_bw=self.edge_bw
        )

    def test_topology_structure(self):
        # Verify the number of core switches
        core_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'core']
        self.assertEqual(len(core_switches), self.num_core_switches)

        # Verify the number of aggregation switches
        agg_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'aggregation']
        self.assertEqual(len(agg_switches), self.num_aggregation_switches)

        # Verify the number of edge switches
        edge_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'edge']
        self.assertEqual(len(edge_switches), self.num_edge_switches)

        # Verify the number of hosts
        hosts = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'host']
        expected_hosts = self.num_edge_switches * self.hosts_per_edge_switch
        self.assertEqual(len(hosts), expected_hosts)

    def test_bandwidth_properties(self):
        # Verify bandwidth of core switches
        for node, data in self.topology.G.nodes(data=True):
            if data['type'] == 'core':
                self.assertEqual(data['bandwidth'], self.core_bw)

        # Verify bandwidth of aggregation switches
        for node, data in self.topology.G.nodes(data=True):
            if data['type'] == 'aggregation':
                self.assertEqual(data['bandwidth'], self.agg_bw)

        # Verify bandwidth of edge switches
        for node, data in self.topology.G.nodes(data=True):
            if data['type'] == 'edge':
                self.assertEqual(data['bandwidth'], self.edge_bw)

    def test_edge_connections(self):
        # Verify core-to-aggregation connections
        core_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'core']
        agg_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'aggregation']

        for core in core_switches:
            connected_agg = [neighbor for neighbor in self.topology.G.neighbors(core)
                             if self.topology.G.nodes[neighbor]['type'] == 'aggregation']
            self.assertGreaterEqual(len(connected_agg), 1)

        # Verify aggregation-to-edge connections
        edge_switches = [node for node, data in self.topology.G.nodes(data=True) if data['type'] == 'edge']
        for agg in agg_switches:
            connected_edges = [neighbor for neighbor in self.topology.G.neighbors(agg)
                               if self.topology.G.nodes[neighbor]['type'] == 'edge']
            self.assertGreaterEqual(len(connected_edges), 1)

        # Verify edge-to-host connections
        for edge in edge_switches:
            connected_hosts = [neighbor for neighbor in self.topology.G.neighbors(edge)
                               if self.topology.G.nodes[neighbor]['type'] == 'host']
            self.assertEqual(len(connected_hosts), self.hosts_per_edge_switch)

    def test_adjacency_matrix(self):
        # Verify adjacency matrix dimensions and values
        adj_matrix = self.topology.calculate_host_to_host_adjacency_matrix()
        num_hosts = self.num_edge_switches * self.hosts_per_edge_switch
        self.assertEqual(adj_matrix.shape, (num_hosts, num_hosts))
        self.assertTrue((adj_matrix >= 0).all())

if __name__ == "__main__":
    unittest.main()
