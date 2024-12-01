import pytest
from fat_tree_topology import FatTreeTopology


def test_allocation_and_deallocation():
    """
    Test allocation and deallocation of bandwidth between nodes.
    """
    topology = FatTreeTopology(num_pods=4, bandwidth=10)

    # Perform single allocation
    node = "edge_0"
    workers = ["host_0", "host_1"]
    allocation = topology.allocate_ps_to_workers_single(node, workers)

    # Check allocations
    for worker in workers:
        assert allocation[worker] == node, f"{worker} not allocated correctly."

    # Check bandwidth after allocation
    for worker in workers:
        assert topology.G[node][worker]['available_bandwidth'] == 0, \
            f"Bandwidth not reduced correctly for {worker}."

    # Deallocate and verify bandwidth restoration
    topology.deallocate_bandwidth(node, workers)  # Updated method call
    for worker in workers:
        assert topology.G[node][worker]['available_bandwidth'] == 10, \
            f"Bandwidth not restored correctly for {worker}."


def test_over_allocation():
    """
    Test handling of over-allocation where nodes request more bandwidth than available.
    """
    topology = FatTreeTopology(num_pods=4, bandwidth=10)

    # Allocate all available bandwidth
    node = "edge_0"
    workers = ["host_0", "host_1"]
    topology.allocate_ps_to_workers_single(node, workers)

    # Attempt to allocate another worker (should fail due to insufficient bandwidth)
    new_worker = ["host_2"]
    allocation = topology.allocate_ps_to_workers_single(node, new_worker)
    assert allocation["host_2"] is None, "Worker allocated despite insufficient bandwidth."


def test_no_available_path():
    """
    Test allocation where no direct path exists between node and workers.
    """
    topology = FatTreeTopology(num_pods=4, bandwidth=10)

    # Remove an edge to simulate no available path
    topology.G.remove_edge("edge_0", "host_0")

    node = "edge_0"
    workers = ["host_0"]
    allocation = topology.allocate_ps_to_workers_single(node, workers)

    # Worker should not be allocated due to missing path
    assert allocation["host_0"] is None, "Worker allocated despite no available path."


def test_allocate_ps_to_workers_balanced():
    """
    Test balanced allocation of workers across nodes.
    """
    topology = FatTreeTopology(num_pods=4, bandwidth=10)

    allocation = topology.allocate_ps_to_workers_balanced()

    # Verify each host is allocated to the best edge switch
    for host, switch in allocation.items():
        assert switch is not None, f"{host} was not allocated to any switch."
        assert topology.G[host][switch]['available_bandwidth'] == 0, \
            f"Bandwidth not correctly reduced for {host}."


def test_allocate_ps_to_workers_single():
    """
    Test single allocation of workers.
    """
    topology = FatTreeTopology(num_pods=4, bandwidth=10)

    node = "edge_0"
    workers = ["host_0", "host_1"]
    allocation = topology.allocate_ps_to_workers_single(node, workers)

    # Verify allocation is correct
    for worker in workers:
        assert allocation[worker] == node, f"{worker} not correctly allocated."
        assert topology.G[node][worker]['available_bandwidth'] == 0, \
            f"Bandwidth not correctly reduced for {worker}."
