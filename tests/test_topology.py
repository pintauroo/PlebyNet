# tests/test_topology.py

import unittest
from src.topology_nx import FatTreeTopology

class TestFatTreeTopology(unittest.TestCase):
    def test_topology_creation(self):
        k = 4  # Must be even
        bandwidth = 100
        topology = FatTreeTopology(k=k, bandwidth=bandwidth)
        G = topology.G

        # Check number of nodes
        expected_core_switches = (k // 2) ** 2
        expected_agg_switches = k * (k // 2)
        expected_edge_switches = k * (k // 2)
        expected_hosts = (k ** 3) // 4
        expected_total_nodes = expected_core_switches + expected_agg_switches + expected_edge_switches + expected_hosts

        self.assertEqual(len(G.nodes), expected_total_nodes)

    def test_bandwidth_allocation(self):
        k = 4  # Must be even
        bandwidth = 100
        topology = FatTreeTopology(k=k, bandwidth=bandwidth)

        # Select some hosts for testing
        allocation_list = [f"H0_0_{i}" for i in range(2)]  # Example hosts
        total_bw = 50
        job_id = 1
        time_instant = 0

        # Allocate bandwidth
        allocated_bw = topology.allocate_bandwidth_between_workers_and_ps(
            allocation_list, total_bw, job_id, time_instant
        )
        self.assertGreater(allocated_bw, 0, "Bandwidth allocation failed when it should have succeeded.")

        # Deallocate bandwidth
        deallocated = topology.deallocate_bandwidth_between_workers_and_ps(job_id, time_instant + 1)
        self.assertTrue(deallocated, "Bandwidth deallocation failed when it should have succeeded.")

        # Verify network state is back to original
        self.assertTrue(topology.verify_network_state(), "Network state did not return to original after deallocation.")

    def test_invalid_k_value(self):
        with self.assertRaises(ValueError):
            FatTreeTopology(k=5, bandwidth=100)  # k must be even

    def test_no_path_between_hosts(self):
        k = 4
        bandwidth = 100
        topology = FatTreeTopology(k=k, bandwidth=bandwidth)

        # Remove some edges to create a partitioned network
        topology.G.remove_edge("E0_0", "A0_0")
        topology.G.remove_edge("E0_0", "A0_1")

        allocation_list = [f"H0_0_0", f"H1_0_0"]  # Hosts that may now be disconnected
        total_bw = 50
        job_id = 2
        time_instant = 0

        allocated_bw = topology.allocate_bandwidth_between_workers_and_ps(
            allocation_list, total_bw, job_id, time_instant
        )
        self.assertEqual(allocated_bw, 0.0, "Bandwidth should not be allocated when no path exists.")

if __name__ == '__main__':
    unittest.main()
