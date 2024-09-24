from typing import List, Tuple
import random
import matplotlib.pyplot as plt
import numpy as np
from src.node import node

class BandwidthAllocationError(Exception):
    """Custom exception for bandwidth allocation issues."""
    pass

# class Node:
#     def __init__(self, node_id: int, max_bw: float):
#         """
#         Initialize a Node.

#         :param node_id: Unique identifier for the node.
#         :param max_bw: Maximum bandwidth capacity of the node.
#         """
#         self.node_id = node_id
#         self.max_bw = max_bw
#         self.allocated_bw = 0.0  # Currently allocated bandwidth
#         self.communication_log = []  # To track communications with other nodes

#     def allocate_bandwidth(self, bw: float):
#         """
#         Allocate bandwidth to the node.

#         :param bw: Bandwidth to allocate.
#         :raises BandwidthAllocationError: If allocation exceeds node capacity.
#         """
#         if self.allocated_bw + bw > self.max_bw:
#             raise BandwidthAllocationError(
#                 f"Node {self.node_id}: Allocation of {bw} exceeds maximum bandwidth {self.max_bw}."
#             )
#         self.allocated_bw += bw
#         # print(f"Node {self.node_id}: Allocated {bw} BW (Total Allocated: {self.allocated_bw}/{self.max_bw})")

#     def __repr__(self):
#         return f"Node(id={self.node_id}, allocated_bw={self.allocated_bw}/{self.max_bw})"

class Switch:
    def __init__(self, switch_id: int, max_capacity: float):
        self.switch_id = switch_id
        self.max_capacity = max_capacity
        self.allocated_capacity = 0.0  # Currently allocated capacity

    def allocate_bandwidth(self, bw: float):
        if self.allocated_capacity + bw > self.max_capacity:
            raise BandwidthAllocationError(
                f"Switch {self.switch_id}: Allocation of {bw} exceeds maximum capacity {self.max_capacity}."
            )
        self.allocated_capacity += bw
        # print(f"Switch {self.switch_id}: Allocated {bw} BW (Total Allocated: {self.allocated_capacity}/{self.max_capacity})")

    def __repr__(self):
        return f"Switch(id={self.switch_id}, allocated_capacity={self.allocated_capacity}/{self.max_capacity})"

class SpineSwitch(Switch):
    def __init__(self, switch_id: int, max_capacity: float):
        super().__init__(switch_id, max_capacity)
        self.connected_leaf_switches: List['LeafSwitch'] = []

    def connect_leaf_switch(self, leaf_switch: 'LeafSwitch'):
        self.connected_leaf_switches.append(leaf_switch)
        # print(f"SpineSwitch {self.switch_id}: Connected to LeafSwitch {leaf_switch.switch_id}")

    def __repr__(self):
        return f"SpineSwitch(id={self.switch_id}, allocated_capacity={self.allocated_capacity}/{self.max_capacity})"

class LeafSwitch(Switch):
    def __init__(self, switch_id: int, max_capacity: float):
        super().__init__(switch_id, max_capacity)
        self.connected_nodes: List[node] = []
        self.connected_spine_switches: List[SpineSwitch] = []

    def connect_node(self, node: node):
        self.connected_nodes.append(node)
        # print(f"LeafSwitch {self.switch_id}: Connected to Node {node.node_id}")

    def connect_spine_switch(self, spine_switch: SpineSwitch):
        self.connected_spine_switches.append(spine_switch)
        spine_switch.connect_leaf_switch(self)
        # print(f"LeafSwitch {self.switch_id}: Connected to SpineSwitch {spine_switch.switch_id}")

    def allocate_bw_to_spine(self, bw: float):
        """Allocate bandwidth to all connected spine switches proportionally."""
        num_spine = len(self.connected_spine_switches)
        bw_per_spine = bw / num_spine
        for spine_switch in self.connected_spine_switches:
            spine_switch.allocate_bandwidth(bw_per_spine)

    def __repr__(self):
        return f"LeafSwitch(id={self.switch_id}, allocated_capacity={self.allocated_capacity}/{self.max_capacity})"


class BandwidthTracker:
    """
    A class to track bandwidth allocations between nodes, switches, and paths.
    """
    def __init__(self):
        self.allocations = {}  # { (node1_id, node2_id): bw_allocated }
    
    def add_allocation(self, node1_id: int, node2_id: int, bw: float):
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        if key in self.allocations:
            self.allocations[key] += bw
        else:
            self.allocations[key] = bw
    
    def get_allocation(self, node1_id: int, node2_id: int) -> float:
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        return self.allocations.get(key, 0.0)
    
    def remove_allocation(self, node1_id: int, node2_id: int, bw: float):
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        if key in self.allocations:
            self.allocations[key] -= bw
            if self.allocations[key] <= 0:
                del self.allocations[key]


class SpineLeafTopology:
    def __init__(self):
        self.nodes: List[node] = []
        self.leaf_switches: List[LeafSwitch] = []
        self.spine_switches: List[SpineSwitch] = []
        self.tracker = BandwidthTracker()
        
        

    def init_topology(self, nodes, num_spine_switches: int, num_leaf_switches: int, num_nodes: int,
                      max_spine_capacity: float, max_leaf_capacity: float, max_node_bw: float):
        """
        Initialize the topology by adding spine switches, leaf switches, and nodes, 
        and connecting the nodes to leaf switches and leaf switches to spine switches.

        :param num_spine_switches: Number of spine switches to add.
        :param num_leaf_switches: Number of leaf switches to add.
        :param num_nodes: Number of nodes to add.
        :param max_spine_capacity: Maximum bandwidth capacity for spine switches.
        :param max_leaf_capacity: Maximum bandwidth capacity for leaf switches.
        :param max_node_bw: Maximum bandwidth capacity for each node.
        """
        # Add spine switches
        for spine_id in range(200, 200+num_spine_switches):
            self.add_spine_switch(switch_id=spine_id, max_capacity=max_spine_capacity)

        # Add leaf switches
        # Initialize leaf switches with IDs starting from 0
        for leaf_id in range(100, 100+num_leaf_switches):  # 0-based, starts at 0
            self.add_leaf_switch(switch_id=leaf_id, max_capacity=max_leaf_capacity)



        # Add nodes
        # for node_id in range(1, num_nodes + 1):
        #     self.add_node(node_id=node_id, max_bw=max_node_bw)

        # Connect nodes to leaf switches (distribute nodes across leaf switches)
        self.nodes = nodes
        # print('yoyoyoyo', self.nodes)
        # for node_id in range(num_nodes):  # Adjust for zero-based indexing
        #     leaf_id = node_id // (num_nodes // num_leaf_switches)  # This should result in valid leaf_id from 0 to num_leaf_switches-1
        #     self.connect_node_to_leaf(node_id=node_id, leaf_id=leaf_id)
        for node_id in range(num_nodes):
            leaf_id = 100 + (node_id) // (num_nodes // num_leaf_switches)  # Assign nodes to leaf switches
            self.connect_node_to_leaf(node_id=node_id, leaf_id=leaf_id)



        # Connect each leaf switch to all spine switches
        # for leaf_id in range(num_leaf_switches):
        #     for spine_id in range(num_spine_switches):
        #         self.connect_leaf_to_spine(leaf_id=leaf_id, spine_id=spine_id)
        for leaf_id in range(100, 100 + num_leaf_switches):
            for spine_id in range(200, 200 + num_spine_switches):
                self.connect_leaf_to_spine(leaf_id=leaf_id, spine_id=spine_id)

        # print("\n--- Topology Setup Complete ---\n")

    def get_node_by_id(self, node_id: int) -> node:
        """
        Retrieve a Node object given its ID.

        :param node_id: The ID of the node.
        :return: The Node object corresponding to the ID.
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"Node with ID {node_id} not found.")

    def allocate_bandwidth(self, ps_node_ids, worker_node_ids, bw_per_worker):
        """
        Allocate bandwidth for all jobs, considering the cumulative effect of all worker-to-PS allocations.
        This ensures bandwidth is checked holistically before committing to allocation.

        If allocation fails, undo any successful allocations made.
        
        :param ps_node_ids: List of parameter server node IDs.
        :param worker_node_ids: List of worker node IDs.
        :param bw_per_worker: Bandwidth required per worker.
        """
        successful_allocations = []  # Track successful allocations for rollback

        # Track if we can allocate bandwidth for all pairs
        can_allocate = True
        remaining_bw = {}  # Dictionary to track remaining bandwidth for each node and switch
        
        # Initialize remaining bandwidth for each PS and Worker node, leaf and spine switches
        for ps_node_id in ps_node_ids:
            ps_node = self.get_node_by_id(ps_node_id)
            remaining_bw[ps_node.node_id] = ps_node.max_bw - ps_node.allocated_bw

        for worker_node_id in worker_node_ids:
            worker_node = self.get_node_by_id(worker_node_id)
            remaining_bw[worker_node.node_id] = worker_node.max_bw - worker_node.allocated_bw

        # Iterate over all worker-to-PS pairs and simulate bandwidth allocation
        for worker_node_id in worker_node_ids:
            worker_node = self.get_node_by_id(worker_node_id)
            for ps_node_id in ps_node_ids:
                ps_node = self.get_node_by_id(ps_node_id)
                
                # Check if this worker-to-PS link can handle the required bandwidth
                if not self.can_simulate_bandwidth_allocation(worker_node, ps_node, bw_per_worker, remaining_bw):
                    can_allocate = False
                    self.rollback_allocations(successful_allocations)  # Rollback any successful allocations
                    return False

        # If all allocations are valid, proceed with the actual allocation
        if can_allocate:
            for worker_node_id in worker_node_ids:
                worker_node = self.get_node_by_id(worker_node_id)
                for ps_node_id in ps_node_ids:
                    ps_node = self.get_node_by_id(ps_node_id)
                    try:
                        # Route traffic and track successful allocation
                        self.route_traffic(worker_node, ps_node, bw_per_worker)
                        self.tracker.add_allocation(worker_node.node_id, ps_node.node_id, bw_per_worker)
                        successful_allocations.append((worker_node.node_id, ps_node.node_id, bw_per_worker))  # Track the successful allocation

                    except BandwidthAllocationError as e:
                        print(f"BandwidthAllocationError: {e}")
                        self.rollback_allocations(successful_allocations)  # Rollback any successful allocations
                        return False

            return True
        else:
            self.rollback_allocations(successful_allocations)  # Rollback in case of failure
            return False
        
    def rollback_allocations(self, allocations):
        """
        Rollback the bandwidth allocations in case of a failure.

        :param allocations: A list of tuples (worker_node_id, ps_node_id, bw_allocated)
        """
        for worker_node_id, ps_node_id, bw_allocated in allocations:
            worker_node = self.get_node_by_id(worker_node_id)
            ps_node = self.get_node_by_id(ps_node_id)

            # Rollback the allocated bandwidth
            self.route_traffic_deallocation(worker_node, ps_node, bw_allocated)
            self.tracker.remove_allocation(worker_node.node_id, ps_node.node_id, bw_allocated)
            print(f"Rolled back {bw_allocated} Mbps between Worker {worker_node_id} and PS {ps_node_id}.")



    def can_simulate_bandwidth_allocation(self, node1: node, node2: node, bw: float, remaining_bw: dict) -> bool:
        """
        Simulate bandwidth allocation between two nodes without committing to the allocation.
        This checks if the allocation is feasible based on the remaining bandwidth.
        
        :param node1: First node.
        :param node2: Second node.
        :param bw: Bandwidth required for the allocation.
        :param remaining_bw: Dictionary tracking the remaining bandwidth of each node and switch.
        :return: True if the allocation is possible, False otherwise.
        """
        if node1 == node2:
            # print('no need to allcate same node!')
            return True
        
        # print()
        # print(f"Simulating bandwidth allocation between node {node1.node_id} and node {node2.node_id} for {bw} bandwidth.")
        
        leaf_switch_1 = self.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.find_leaf_switch_for_node(node2)
        
        # print(f"Leaf switch for node {node1.node_id}: {leaf_switch_1.switch_id}, for node {node2.node_id}: {leaf_switch_2.switch_id}")
        
        # Check node bandwidth availability
        # print(f"Remaining bandwidth for node {node1.node_id}: {remaining_bw[node1.node_id]}, node {node2.node_id}: {remaining_bw[node2.node_id]}")
        if remaining_bw[node1.node_id] < bw:
            # print(f"Node {node1.node_id} does not have enough bandwidth. Required: {bw}, Available: {remaining_bw[node1.node_id]}")
            return False
        if remaining_bw[node2.node_id] < bw:
            # print(f"Node {node2.node_id} does not have enough bandwidth. Required: {bw}, Available: {remaining_bw[node2.node_id]}")
            return False
        
        # Check leaf switch bandwidth availability
        # print(f"Remaining bandwidth for leaf switch {leaf_switch_1.switch_id}: {remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity)}")
        # print(f"Remaining bandwidth for leaf switch {leaf_switch_2.switch_id}: {remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity)}")
        
        if remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity) < bw:
            # print(f"Leaf switch {leaf_switch_1.switch_id} does not have enough bandwidth. Required: {bw}, Available: {remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity)}")
            return False
        else:
            # print(f"Leaf switch {leaf_switch_1.switch_id} HAS enough bandwidth. Required: {bw}, Available: {remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity)}")
            pass
        if remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity) < bw:
            # print(f"Leaf switch {leaf_switch_2.switch_id} does not have enough bandwidth. Required: {bw}, Available: {remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity)}")
            return False
        else:
            # print(f"Leaf switch {leaf_switch_2.switch_id} HAS enough bandwidth. Required: {bw}, Available: {remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity)}")
            pass
        # Check spine switch bandwidth if they are on different leaf switches
        if leaf_switch_1 != leaf_switch_2:
            # print(f"Nodes are on different leaf switches. Checking spine switches...")
            common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
            # print(f"Common spine switches: {[spine.switch_id for spine in common_spines]}")
            
            if not common_spines:
                # print(f"No common spine switches found between leaf switch {leaf_switch_1.switch_id} and leaf switch {leaf_switch_2.switch_id}.")
                return False
            
            for spine_switch in common_spines:
                # print(f"Remaining bandwidth for spine switch {spine_switch.switch_id}: {remaining_bw.get(spine_switch.switch_id, spine_switch.max_capacity)}")
                if remaining_bw.get(spine_switch.switch_id, spine_switch.max_capacity) < bw / len(common_spines):
                    # print(f"Spine switch {spine_switch.switch_id} does not have enough bandwidth. Required: {bw / len(common_spines)}, Available: {remaining_bw.get(spine_switch.switch_id, spine_switch.max_capacity)}")
                    return False

        # Simulate bandwidth reduction after the allocation
        # print(f"Simulating bandwidth allocation: reducing bandwidth for nodes and switches.")
        remaining_bw[node1.node_id] -= bw
        remaining_bw[node2.node_id] -= bw
        # print(f"Remaining bandwidth after reduction for node {node1.node_id}: {remaining_bw[node1.node_id]}, node {node2.node_id}: {remaining_bw[node2.node_id]}")
        
        remaining_bw[leaf_switch_1.switch_id] = remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity)
        remaining_bw[leaf_switch_2.switch_id] = remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity)
        # print(f"BEFORE!!! Remaining bandwidth after reduction for leaf switch {leaf_switch_1.switch_id}: {remaining_bw[leaf_switch_1.switch_id]}, leaf switch {leaf_switch_2.switch_id}: {remaining_bw[leaf_switch_2.switch_id]}")
       

        remaining_bw[leaf_switch_1.switch_id] = remaining_bw.get(leaf_switch_1.switch_id, leaf_switch_1.max_capacity) - bw
        remaining_bw[leaf_switch_2.switch_id] = remaining_bw.get(leaf_switch_2.switch_id, leaf_switch_2.max_capacity) - bw
        # print(f"AFTER!!! Remaining bandwidth after reduction for leaf switch {leaf_switch_1.switch_id}: {remaining_bw[leaf_switch_1.switch_id]}, leaf switch {leaf_switch_2.switch_id}: {remaining_bw[leaf_switch_2.switch_id]}")
        
        if leaf_switch_1 != leaf_switch_2:
            for spine_switch in common_spines:
                remaining_bw[spine_switch.switch_id] = remaining_bw.get(spine_switch.switch_id, spine_switch.max_capacity) - (bw / len(common_spines))
                # print(f"Remaining bandwidth after reduction for spine switch {spine_switch.switch_id}: {remaining_bw[spine_switch.switch_id]}")
        
        return True

    def route_traffic(self, node1_id: int, node2_id: int, bw: float):
        """
        Automatically find the path between two nodes (identified by IDs) and allocate bandwidth along the path.

        :param node1_id: ID of the source node.
        :param node2_id: ID of the destination node.
        :param bw: Bandwidth to allocate for the communication.
        """
        node1 = self.get_node_by_id(node1_id)
        node2 = self.get_node_by_id(node2_id)
        if node1 != node2:

            # Find leaf switches for both nodes
            leaf_switch_1 = self.find_leaf_switch_for_node(node1)
            leaf_switch_2 = self.find_leaf_switch_for_node(node2)

            if not leaf_switch_1 or not leaf_switch_2:
                raise ValueError(f"Routing Error: Unable to find leaf switches for nodes {node1.node_id} and {node2.node_id}.")

            # If both nodes are on the same leaf switch
            if leaf_switch_1 == leaf_switch_2:
                # print(f"Both Node {node1.node_id} and Node {node2.node_id} are on LeafSwitch {leaf_switch_1.switch_id}.")
                self.allocate_bandwidth_along_path(node1, node2, leaf_switch_1, bw)
            else:
                # If nodes are on different leaf switches, use spine switches
                # print(f"Node {node1.node_id} is on LeafSwitch {leaf_switch_1.switch_id}, Node {node2.node_id} is on LeafSwitch {leaf_switch_2.switch_id}.")
                self.allocate_bandwidth_with_spine(node1, node2, leaf_switch_1, leaf_switch_2, bw)

    def can_allocate_bandwidth(self, node1: node, node2: node, bw: float) -> bool:
        """
        Check if the bandwidth can be allocated between two nodes without exceeding capacities.
        
        :param node1: First node.
        :param node2: Second node.
        :param bw: Requested bandwidth to check.
        :return: True if the bandwidth can be allocated, False otherwise.
        """
        # Check bandwidth on the nodes
        if node1.allocated_bw + bw > node1.max_bw or node2.allocated_bw + bw > node2.max_bw:
            return False

        # Check bandwidth on the switches (leaf and spine)
        leaf_switch_1 = self.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.find_leaf_switch_for_node(node2)

        if leaf_switch_1.allocated_capacity + bw > leaf_switch_1.max_capacity or \
           leaf_switch_2.allocated_capacity + bw > leaf_switch_2.max_capacity:
            return False

        # If nodes are on different leaf switches, check spine switches
        if leaf_switch_1 != leaf_switch_2:
            common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
            if not common_spines:
                return False
            for spine_switch in common_spines:
                if spine_switch.allocated_capacity + (bw / len(common_spines)) > spine_switch.max_capacity:
                    return False

        return True

    def get_adjacency_matrix(self):
        """
        Create an adjacency matrix representing the available bandwidth between each pair of nodes.
        The value at [i][j] represents the available bandwidth between Node i and Node j.
        If there is no direct connection, the value will be 0.

        :return: A 2D numpy array representing the adjacency matrix of available bandwidth.
        """
        num_nodes = len(self.nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        # # print('getting adj')

        # Iterate over all node pairs and check the available bandwidth between them
        for i in range(num_nodes):  # Starts at 0
            for j in range(i + 1, num_nodes):  # Adjust for zero-based index
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                available_bw = self.get_available_bandwidth_between_nodes(node1, node2)
                adjacency_matrix[i][j] = available_bw
                adjacency_matrix[j][i] = available_bw  # Symmetric matrix

        return adjacency_matrix
    
    def get_available_bandwidth_between_nodes(self, node1: node, node2: node) -> float:
        """
        Calculate the available bandwidth between two nodes.
        This function assumes that bandwidth is routed through leaf and spine switches,
        and takes into account the maximum node bandwidth.

        :param node1: First node.
        :param node2: Second node.
        :return: The available bandwidth between node1 and node2.
        """
        leaf_switch_1 = self.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.find_leaf_switch_for_node(node2)

        # If both nodes are on the same leaf switch, return the available bandwidth on the leaf switch
        if leaf_switch_1 == leaf_switch_2:
            available_bw = min(leaf_switch_1.max_capacity - leaf_switch_1.allocated_capacity,
                            node1.max_bw - node1.allocated_bw,
                            node2.max_bw - node2.allocated_bw)
            return available_bw

        # If nodes are on different leaf switches, calculate the available bandwidth through the spine switches
        # # print()
        common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
        if not common_spines:
            return 0.0  # No direct connection between nodes

        # Find the minimum available bandwidth on all common spines
        spine_bw = min(spine_switch.max_capacity - spine_switch.allocated_capacity for spine_switch in common_spines)

        # The available bandwidth is constrained by the smallest available capacity along the path
        leaf1_available_bw = leaf_switch_1.max_capacity - leaf_switch_1.allocated_capacity
        leaf2_available_bw = leaf_switch_2.max_capacity - leaf_switch_2.allocated_capacity

        # Take into account the node bandwidth (node1.max_bw and node2.max_bw)
        node1_available_bw = node1.max_bw - node1.allocated_bw
        node2_available_bw = node2.max_bw - node2.allocated_bw

        # The available bandwidth is the minimum of the available capacities along the path and the node bandwidths
        return min(leaf1_available_bw, leaf2_available_bw, spine_bw, node1_available_bw, node2_available_bw)

    def print_adjacency_matrix(self, adjacency_matrix):
        """
        Pretty-print the adjacency matrix.

        :param adjacency_matrix: The 2D numpy array representing the adjacency matrix.
        """
        # print("Adjacency Matrix (Available Bandwidth Between Nodes):")
        # print(adjacency_matrix)
    
    def plot_spine_utilization(self):
        """
        Plot the bandwidth utilization of all spine switches as a percentage of their total capacity.
        """
        spine_switch_ids = [spine.switch_id for spine in self.spine_switches]
        spine_bw_utilization = [(spine.allocated_capacity / spine.max_capacity) * 100 for spine in self.spine_switches]  # Utilization in percentage

        # Plotting the spine switch bandwidth utilization
        plt.figure(figsize=(10, 5))
        plt.bar(spine_switch_ids, spine_bw_utilization, color='red')
        plt.xlabel('Spine Switch ID')
        plt.ylabel('Bandwidth Utilization (%)')
        plt.title('Spine Switch Bandwidth Utilization')
        plt.savefig('spine.png')
    
    def plot_congestion_metric(self):
        """
        Plot the congestion metric: Total bandwidth requested vs total available bandwidth.
        """
        node_ids = [node.node_id for node in self.nodes]
        total_bw_requested = [node.allocated_bw for node in self.nodes]
        total_bw_available = [node.max_bw for node in self.nodes]
        congestion = [requested - available for requested, available in zip(total_bw_requested, total_bw_available)]

        # Plotting the congestion metric (over-allocated bandwidth)
        plt.figure(figsize=(10, 5))
        plt.bar(node_ids, congestion, color='orange')
        plt.xlabel('Node ID')
        plt.ylabel('Over-Allocated Bandwidth (Mbps)')
        plt.title('Bandwidth Congestion (Over-Allocated Bandwidth per Node)')
        plt.savefig('over.png')
    
    def plot_utilization_metric(self):
        """
        Plot the bandwidth utilization per node.
        """
        node_ids = [node.node_id for node in self.nodes]
        utilization = [(node.allocated_bw / node.max_bw) * 100 for node in self.nodes]  # Utilization in percentage

        # Plotting the utilization metric (Bandwidth utilization percentage)
        plt.figure(figsize=(10, 5))
        plt.bar(node_ids, utilization, color='orange')
        plt.xlabel('Node ID')
        plt.ylabel('Bandwidth Utilization (%)')
        plt.title('Bandwidth Utilization per Node')
        plt.savefig('util.png')

    def plot_bandwidth_utilization(self):
        """
        Plot the bandwidth utilization of all nodes, leaf switches, and spine switches.
        """
        # Gather data
        node_ids = [node.node_id for node in self.nodes]
        node_bw_utilization = [(node.allocated_bw / node.max_bw) * 100 for node in self.nodes]  # Utilization in percentage

        leaf_switch_ids = [leaf.switch_id for leaf in self.leaf_switches]
        leaf_bw_utilization = [(leaf.allocated_capacity / leaf.max_capacity) * 100 for leaf in self.leaf_switches]  # Utilization in percentage

        spine_switch_ids = [spine.switch_id for spine in self.spine_switches]
        spine_bw_utilization = [(spine.allocated_capacity / spine.max_capacity) * 100 for spine in self.spine_switches]  # Utilization in percentage

        # Plotting node bandwidth utilization
        plt.figure(figsize=(10, 5))
        plt.bar(node_ids, node_bw_utilization, color='blue')
        plt.xlabel('Node ID')
        plt.ylabel('Bandwidth Utilization (%)')
        plt.title('Node Bandwidth Utilization')
        plt.savefig('bw.png')

        # Plotting leaf switch bandwidth utilization
        plt.figure(figsize=(10, 5))
        plt.bar(leaf_switch_ids, leaf_bw_utilization, color='green')
        plt.xlabel('Leaf Switch ID')
        plt.ylabel('Bandwidth Utilization (%)')
        plt.title('Leaf Switch Bandwidth Utilization')
        plt.savefig('topology.png')

    def add_node(self, node_id: int, max_bw: float):
        node = node(node_id, max_bw)
        self.nodes.append(node)
        # print(f"Topology: Added {node}")
    
    def add_leaf_switch(self, switch_id: int, max_capacity: float):
        leaf = LeafSwitch(switch_id, max_capacity)
        self.leaf_switches.append(leaf)
        # print(f"Topology: Added {leaf}")

    def add_spine_switch(self, switch_id: int, max_capacity: float):
        spine = SpineSwitch(switch_id, max_capacity)
        self.spine_switches.append(spine)
        # print(f"Topology: Added {spine}")

    def connect_node_to_leaf(self, node_id: int, leaf_id: int):
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        leaf = next((l for l in self.leaf_switches if l.switch_id == leaf_id), None)
        if not node:
            raise ValueError(f"Connection Error: Node {node_id} not found.")
        if not leaf:
            raise ValueError(f"Connection Error: LeafSwitch {leaf_id} not found.")
        leaf.connect_node(node)

    def connect_leaf_to_spine(self, leaf_id: int, spine_id: int):
        leaf = next((l for l in self.leaf_switches if l.switch_id == leaf_id), None)
        spine = next((s for s in self.spine_switches if s.switch_id == spine_id), None)
        
        if not leaf:
            # print(f"Available leaf switches: {[l.switch_id for l in self.leaf_switches]}")
            raise ValueError(f"Connection Error: LeafSwitch {leaf_id} not found.")
        
        if not spine:
            # print(f"Available spine switches: {[s.switch_id for s in self.spine_switches]}")
            raise ValueError(f"Connection Error: SpineSwitch {spine_id} not found.")
        
        leaf.connect_spine_switch(spine)

    def route_traffic(self, node1: node, node2: node, bw: float):
        """
        Automatically find the path between two nodes and allocate bandwidth along the path.

        :param node1: Source node.
        :param node2: Destination node.
        :param bw: Bandwidth to allocate for the communication.
        """
        # Find leaf switches for both nodes
        leaf_switch_1 = self.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.find_leaf_switch_for_node(node2)

        if not leaf_switch_1 or not leaf_switch_2:
            raise ValueError(f"Routing Error: Unable to find leaf switches for nodes {node1.node_id} and {node2.node_id}.")

        # If both nodes are on the same leaf switch
        if leaf_switch_1 == leaf_switch_2:
            # print(f"Both Node {node1.node_id} and Node {node2.node_id} are on LeafSwitch {leaf_switch_1.switch_id}.")
            self.allocate_bandwidth_along_path(node1, node2, leaf_switch_1, bw)
        else:
            # If nodes are on different leaf switches, use spine switches
            # print(f"Node {node1.node_id} is on LeafSwitch {leaf_switch_1.switch_id}, Node {node2.node_id} is on LeafSwitch {leaf_switch_2.switch_id}.")
            self.allocate_bandwidth_with_spine(node1, node2, leaf_switch_1, leaf_switch_2, bw)

    def find_leaf_switch_for_node(self, node: node) -> LeafSwitch:
        """
        Find the leaf switch to which a node is connected.

        :param node: Node whose leaf switch is being found.
        :return: The leaf switch the node is connected to, or None if not found.
        """
        for leaf in self.leaf_switches:
            if node in leaf.connected_nodes:
                return leaf
        return None

    def allocate_bandwidth_along_path(self, node1: node, node2: node, leaf_switch: LeafSwitch, bw: float):
        """
        Allocate bandwidth for both nodes along the path on the same leaf switch.

        :param node1: Source node.
        :param node2: Destination node.
        :param leaf_switch: The leaf switch both nodes are connected to.
        :param bw: Bandwidth to allocate for the communication.
        """
        # Allocate bandwidth on the nodes
        if node1 != node2:
            node1.allocate_bandwidth(bw)
            node2.allocate_bandwidth(bw)

            # Allocate bandwidth on the leaf switch
            leaf_switch.allocate_bandwidth(bw)

    def allocate_bandwidth_with_spine(self, node1: node, node2: node, leaf_switch_1: LeafSwitch, leaf_switch_2: LeafSwitch, bw: float):
        """
        Allocate bandwidth for both nodes along the path using spine switches.

        :param node1: Source node.
        :param node2: Destination node.
        :param leaf_switch_1: Leaf switch of the source node.
        :param leaf_switch_2: Leaf switch of the destination node.
        :param bw: Bandwidth to allocate for the communication.
        """
        # Allocate bandwidth on both nodes
        node1.allocate_bandwidth(bw)
        node2.allocate_bandwidth(bw)

        # Allocate bandwidth on both leaf switches
        leaf_switch_1.allocate_bandwidth(bw)
        leaf_switch_2.allocate_bandwidth(bw)

        # Find the common spine switches
        common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
        if not common_spines:
            raise ValueError(f"No common spine switches between LeafSwitch {leaf_switch_1.switch_id} and LeafSwitch {leaf_switch_2.switch_id}.")

        # Allocate bandwidth on the common spine switches
        for spine_switch in common_spines:
            spine_switch.allocate_bandwidth(bw / len(common_spines))

    def assert_allocation(self, node1: node, node2: node, bw: float):
        """
        Test if the bandwidth allocation between two nodes is correct.

        :param node1: Source node.
        :param node2: Destination node.
        :param bw: Expected bandwidth to allocate.
        """
        assert node1.allocated_bw >= bw, f"Node {node1.node_id} does not have enough allocated bandwidth."
        assert node2.allocated_bw >= bw, f"Node {node2.node_id} does not have enough allocated bandwidth."
        # print(f"Test passed: Node {node1.node_id} and Node {node2.node_id} have correctly allocated {bw} bandwidth.")

    def reset_allocations(self):
        """
        Reset the allocated bandwidth for all nodes and switches.
        """
        for node in self.nodes:
            node.allocated_bw = 0.0
        for leaf in self.leaf_switches:
            leaf.allocated_capacity = 0.0
        for spine in self.spine_switches:
            spine.allocated_capacity = 0.0
    
    def deallocate_bandwidth(self, ps_node_ids, worker_node_ids, bw_per_worker):
        """
        Deallocate bandwidth for all worker-to-PS pairs based on the actual allocations made.
        """
        # print(f"\nDeallocating bandwidth for Job")

        # Iterate over all worker-to-PS pairs and reverse the allocation
        for worker_node_id in worker_node_ids:
            worker_node = self.get_node_by_id(worker_node_id)
            for ps_node_id in ps_node_ids:
                ps_node = self.get_node_by_id(ps_node_id)

                # Get the actual bandwidth allocated for this worker-PS pair
                if worker_node.node_id == ps_node.node_id:
                    pass
                else:
                    allocated_bw = self.tracker.get_allocation(worker_node.node_id, ps_node.node_id)
                    if allocated_bw > 0:
                        deallocation_bw = min(allocated_bw, bw_per_worker)  # Deallocate only the amount allocated
                        self.route_traffic_deallocation(worker_node, ps_node, deallocation_bw)
                        # Update the tracker with the deallocated bandwidth
                        self.tracker.remove_allocation(worker_node.node_id, ps_node.node_id, deallocation_bw)

        # print(f"Bandwidth deallocation complete for all Worker-PS pairs.")

    def route_traffic_deallocation(self, node1: node, node2: node, bw: float, eps=1e-6):
        """
        Deallocate bandwidth between two nodes, ensuring no negative bandwidth is allocated.
        Handle floating-point precision issues by rounding small values.
        
        :param node1: Source node.
        :param node2: Destination node.
        :param bw: Bandwidth to deallocate for the communication.
        :param eps: Small epsilon value to handle floating-point precision.
        """
        # Find leaf switches for both nodes
        leaf_switch_1 = self.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.find_leaf_switch_for_node(node2)

        if not leaf_switch_1 or not leaf_switch_2:
            raise ValueError(f"Deallocation Error: Unable to find leaf switches for nodes {node1.node_id} and {node2.node_id}.")

        # Deallocate bandwidth on the nodes
        deallocated_bw_node1 = min(node1.allocated_bw, bw)
        deallocated_bw_node2 = min(node2.allocated_bw, bw)
        node1.allocated_bw = max(node1.allocated_bw - deallocated_bw_node1, 0)
        node2.allocated_bw = max(node2.allocated_bw - deallocated_bw_node2, 0)
        # print(f"Deallocated {round(deallocated_bw_node1, 2)} Mbps from Node {node1.node_id} and {round(deallocated_bw_node2, 2)} Mbps from Node {node2.node_id}. New allocation: {round(node1.allocated_bw, 2)}/{node1.max_bw} and {round(node2.allocated_bw, 2)}/{node2.max_bw}.")

        # Deallocate bandwidth on the leaf switches
        if leaf_switch_1 == leaf_switch_2:
            # If both nodes are on the same leaf switch, deallocate bandwidth only once
            deallocated_bw_leaf = min(leaf_switch_1.allocated_capacity, bw)
            leaf_switch_1.allocated_capacity = max(leaf_switch_1.allocated_capacity - deallocated_bw_leaf, 0)
            # print(f"Deallocated {round(deallocated_bw_leaf, 2)} Mbps from LeafSwitch {leaf_switch_1.switch_id}. New capacity: {round(leaf_switch_1.allocated_capacity, 2)}/{leaf_switch_1.max_capacity}.")
        else:
            # If nodes are on different leaf switches, deallocate bandwidth separately for both
            deallocated_bw_leaf1 = min(leaf_switch_1.allocated_capacity, bw)
            deallocated_bw_leaf2 = min(leaf_switch_2.allocated_capacity, bw)
            leaf_switch_1.allocated_capacity = max(leaf_switch_1.allocated_capacity - deallocated_bw_leaf1, 0)
            leaf_switch_2.allocated_capacity = max(leaf_switch_2.allocated_capacity - deallocated_bw_leaf2, 0)
            # print(f"Deallocated {round(deallocated_bw_leaf1, 2)} Mbps from LeafSwitch {leaf_switch_1.switch_id} and {round(deallocated_bw_leaf2, 2)} Mbps from LeafSwitch {leaf_switch_2.switch_id}. New capacity: {round(leaf_switch_1.allocated_capacity, 2)}/{leaf_switch_1.max_capacity} and {round(leaf_switch_2.allocated_capacity, 2)}/{leaf_switch_2.max_capacity}.")

        # If nodes are on different leaf switches, deallocate bandwidth on the common spine switches
        if leaf_switch_1 != leaf_switch_2:
            common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
            if not common_spines:
                raise ValueError(f"No common spine switches found between LeafSwitch {leaf_switch_1.switch_id} and LeafSwitch {leaf_switch_2.switch_id}.")

            for spine_switch in common_spines:
                deallocated_bw_spine = min(spine_switch.allocated_capacity, bw / len(common_spines))
                spine_switch.allocated_capacity = max(spine_switch.allocated_capacity - deallocated_bw_spine, 0)

                # Round small values to avoid floating-point errors
                if abs(spine_switch.allocated_capacity) < eps:
                    spine_switch.allocated_capacity = 0
                # print(f"Deallocated {round(deallocated_bw_spine, 2)} Mbps from SpineSwitch {spine_switch.switch_id}. New capacity: {round(spine_switch.allocated_capacity, 2)}/{spine_switch.max_capacity}.")


        def __repr__(self):
            return (f"SpineLeafTopology(\n"
                    f"  Nodes: {self.nodes},\n"
                    f"  Leaf Switches: {self.leaf_switches},\n"
                    f"  Spine Switches: {self.spine_switches}\n"
                    f")")

class JobGenerator:
    def __init__(self, topology: SpineLeafTopology):
        self.topology = topology
        self.jobs = []

    def create_job(self, ps_node_ids: List[int], worker_node_ids: List[int], bw_per_worker: float):
        """
        Create a distributed training job with a parameter server and workers.

        :param ps_node_ids: List of node IDs acting as parameter servers.
        :param worker_node_ids: List of node IDs acting as workers.
        :param bw_per_worker: Bandwidth required per worker to communicate with the parameter server(s).
        """
        ps_nodes = [self.get_node_by_id(node_id) for node_id in ps_node_ids]
        worker_nodes = [self.get_node_by_id(node_id) for node_id in worker_node_ids]
        job = {'ps_nodes': ps_nodes, 'worker_nodes': worker_nodes, 'bw_per_worker': bw_per_worker}
        self.jobs.append(job)
        # print(f"JobGenerator: Created job with PS nodes {ps_node_ids} and worker nodes {worker_node_ids}")

    def get_node_by_id(self, node_id: int) -> node:
        node = next((n for n in self.topology.nodes if n.node_id == node_id), None)
        if not node:
            raise ValueError(f"Node {node_id} not found in the topology.")
        return node
    def can_allocate_bandwidth(self, node1: node, node2: node, bw: float) -> bool:
        """
        Check if the bandwidth can be allocated between two nodes without exceeding capacities.
        
        :param node1: First node.
        :param node2: Second node.
        :param bw: Requested bandwidth to check.
        :return: True if the bandwidth can be allocated, False otherwise.
        """
        # print(f"Checking bandwidth allocation between node {node1.id} and node {node2.id} for {bw} bandwidth.")

        # Check bandwidth on the nodes
        # print(f"Node {node1.id} allocated_bw: {node1.allocated_bw}, max_bw: {node1.max_bw}")
        # print(f"Node {node2.id} allocated_bw: {node2.allocated_bw}, max_bw: {node2.max_bw}")
        
        if node1.allocated_bw + bw > node1.max_bw:
            # print(f"Node {node1.id} cannot allocate additional bandwidth. Requested: {bw}, Available: {node1.max_bw - node1.allocated_bw}")
            return False
        if node2.allocated_bw + bw > node2.max_bw:
            # print(f"Node {node2.id} cannot allocate additional bandwidth. Requested: {bw}, Available: {node2.max_bw - node2.allocated_bw}")
            return False

        # Check bandwidth on the switches (leaf and spine)
        leaf_switch_1 = self.topology.find_leaf_switch_for_node(node1)
        leaf_switch_2 = self.topology.find_leaf_switch_for_node(node2)

        # print(f"Leaf switch for node {node1.id}: {leaf_switch_1.id}, allocated_capacity: {leaf_switch_1.allocated_capacity}, max_capacity: {leaf_switch_1.max_capacity}")
        # print(f"Leaf switch for node {node2.id}: {leaf_switch_2.id}, allocated_capacity: {leaf_switch_2.allocated_capacity}, max_capacity: {leaf_switch_2.max_capacity}")
        
        if leaf_switch_1.allocated_capacity + bw > leaf_switch_1.max_capacity:
            # print(f"Leaf switch {leaf_switch_1.id} cannot allocate additional bandwidth. Requested: {bw}, Available: {leaf_switch_1.max_capacity - leaf_switch_1.allocated_capacity}")
            return False
        if leaf_switch_2.allocated_capacity + bw > leaf_switch_2.max_capacity:
            # print(f"Leaf switch {leaf_switch_2.id} cannot allocate additional bandwidth. Requested: {bw}, Available: {leaf_switch_2.max_capacity - leaf_switch_2.allocated_capacity}")
            return False

        # If nodes are on different leaf switches, check spine switches
        if leaf_switch_1 != leaf_switch_2:
            # print(f"Nodes are on different leaf switches. Checking spine switches...")
            common_spines = set(leaf_switch_1.connected_spine_switches) & set(leaf_switch_2.connected_spine_switches)
            # print(f"Common spine switches: {[spine.id for spine in common_spines]}")
            
            if not common_spines:
                # print(f"No common spine switches found between leaf switch {leaf_switch_1.id} and leaf switch {leaf_switch_2.id}.")
                return False
            
            for spine_switch in common_spines:
                # print(f"Spine switch {spine_switch.id}, allocated_capacity: {spine_switch.allocated_capacity}, max_capacity: {spine_switch.max_capacity}")
                if spine_switch.allocated_capacity + (bw / len(common_spines)) > spine_switch.max_capacity:
                    # print(f"Spine switch {spine_switch.id} cannot allocate additional bandwidth. Requested: {bw / len(common_spines)}, Available: {spine_switch.max_capacity - spine_switch.allocated_capacity}")
                    return False

        # print("Bandwidth allocation is possible.")
        return True


    def allocate_bandwidth_for_jobs(self):
        """
        Allocate bandwidth for all jobs in the job list, ensuring full allocation or none at all.
        """
        for idx, job in enumerate(self.jobs):
            # print(f"\nAllocating bandwidth for Job {idx+1}")
            ps_nodes = job['ps_nodes']
            worker_nodes = job['worker_nodes']
            bw_per_worker = job['bw_per_worker']

            can_allocate = True

            # Check if the bandwidth can be allocated for every worker-PS pair
            for worker_node in worker_nodes:
                for ps_node in ps_nodes:
                    if not self.can_allocate_bandwidth(worker_node, ps_node, bw_per_worker):
                        can_allocate = False
                        # print(f"Job {idx+1}: Cannot allocate {bw_per_worker} Mbps between Worker {worker_node.node_id} and PS {ps_node.node_id}. Skipping allocation.")
                        break
                if not can_allocate:
                    break

            # If all bandwidth can be allocated, proceed with allocation
            if can_allocate:
                for worker_node in worker_nodes:
                    for ps_node in ps_nodes:
                        try:
                            self.topology.route_traffic(worker_node, ps_node, bw_per_worker)
                        except BandwidthAllocationError as e:
                            print(f"BandwidthAllocationError: {e}")
            else:
                # print(f"Job {idx+1}: Skipped due to insufficient bandwidth.")
                pass

    def reset_jobs(self):
        """
        Reset all jobs and clear bandwidth allocations.
        """
        self.jobs = []
        self.topology.reset_allocations()
        # print("JobGenerator: Reset all jobs and bandwidth allocations.")

class RandomJobGenerator(JobGenerator):
    def __init__(self, topology: SpineLeafTopology):
        super().__init__(topology)

    def create_random_jobs_to_saturate_bw(self, num_jobs: int, max_bw_per_worker: float):
        """
        Create random jobs to saturate the bandwidth.

        :param num_jobs: Number of jobs to create.
        :param max_bw_per_worker: Maximum bandwidth to allocate per worker.
        """
        all_node_ids = [node.node_id for node in self.topology.nodes]

        for i in range(num_jobs):
            # Randomly select a parameter server node (1 PS per job)
            ps_node = random.choice(all_node_ids)
            remaining_nodes = [node_id for node_id in all_node_ids if node_id != ps_node]

            # Randomly select 2 to 4 worker nodes for the job
            num_workers = random.randint(2, 4)
            worker_nodes = random.sample(remaining_nodes, num_workers)

            # Randomly assign bandwidth per worker (up to the specified max_bw_per_worker)
            bw_per_worker = random.uniform(10.0, max_bw_per_worker)

            # Create the job with the selected PS, workers, and bandwidth
            self.create_job(ps_node_ids=[ps_node], worker_node_ids=worker_nodes, bw_per_worker=bw_per_worker)

            # print(f"Random Job {i+1}: PS Node {ps_node}, Worker Nodes {worker_nodes}, BW per worker: {bw_per_worker:.2f} Mbps")

class SequentialJobGenerator(JobGenerator):
    def __init__(self, topology: SpineLeafTopology):
        super().__init__(topology)


    def create_random_job(self, ps_node_count: int, worker_node_count: int, bw_per_worker: float):
        """
        Create a distributed training job with a randomly assigned set of parameter servers (PS)
        and workers, ensuring the job is feasible in terms of bandwidth allocation.

        :param ps_node_count: Number of parameter server nodes required.
        :param worker_node_count: Number of worker nodes required.
        :param bw_per_worker: Bandwidth required per worker to communicate with the parameter server(s).
        """
        all_node_ids = [node.node_id for node in self.topology.nodes]
        
        # Randomly select parameter server nodes
        ps_nodes = random.sample(all_node_ids, ps_node_count)
        
        # Remove selected PS nodes from the pool of available nodes for workers
        available_worker_nodes = [node_id for node_id in all_node_ids if node_id not in ps_nodes]
        
        # Randomly select worker nodes
        worker_nodes = random.sample(available_worker_nodes, worker_node_count)

        # Get the actual Node objects for PS and worker nodes
        ps_node_objects = [self.get_node_by_id(node_id) for node_id in ps_nodes]
        worker_node_objects = [self.get_node_by_id(node_id) for node_id in worker_nodes]

        # Add the job to the job list if both PS and worker nodes are found
        job = {'ps_nodes': ps_node_objects, 'worker_nodes': worker_node_objects, 'bw_per_worker': bw_per_worker}
        self.jobs.append(job)
        # print(f"RandomJobGenerator: Created job with PS nodes {ps_nodes} and worker nodes {worker_nodes}, "
            # f"Bandwidth per worker: {bw_per_worker:.2f} Mbps")

    def create_sequential_job(self, ps_node_count: int, worker_node_count: int, bw_per_worker: float):
        """
        Create a distributed training job using a sequential allocation strategy.

        :param ps_node_count: Number of parameter server nodes required.
        :param worker_node_count: Number of worker nodes required.
        :param bw_per_worker: Bandwidth required per worker to communicate with the parameter server(s).
        """
        all_node_ids = [node.node_id for node in self.topology.nodes]
        
        ps_nodes = []
        worker_nodes = []

        # Sequentially find parameter server nodes
        for node_id in all_node_ids:
            node = self.get_node_by_id(node_id)
            if node.allocated_bw + bw_per_worker <= node.max_bw:
                ps_nodes.append(node)
            if len(ps_nodes) >= ps_node_count:
                break
        

        # If enough parameter servers were not found, discard the job
        if len(ps_nodes) < ps_node_count:
            # print(f"Job discarded: Not enough parameter servers found for {ps_node_count} PS nodes.")
            return
        
        # Sequentially find worker nodes
        for node_id in all_node_ids:
            node = self.get_node_by_id(node_id)
            if node.allocated_bw + bw_per_worker <= node.max_bw and node not in ps_nodes:
                worker_nodes.append(node)
            if len(worker_nodes) >= worker_node_count:
                break

        # If enough worker nodes were not found, discard the job
        if len(worker_nodes) < worker_node_count:
            # print(f"Job discarded: Not enough worker nodes found for {worker_node_count} workers.")
            return

        # Add the job to the job list if both PS and worker nodes are found
        job = {'ps_nodes': ps_nodes, 'worker_nodes': worker_nodes, 'bw_per_worker': bw_per_worker}
        self.jobs.append(job)
        # print(f"SequentialJobGenerator: Created job with PS nodes {[node.node_id for node in ps_nodes]} and worker nodes {[node.node_id for node in worker_nodes]}")

    def allocate_bandwidth_for_jobs(self):
        """
        Allocate bandwidth for all jobs in the job list using a sequential allocation strategy.
        """
        for idx, job in enumerate(self.jobs):
            # print(f"\nAllocating bandwidth for Job {idx+1}")
            ps_nodes = job['ps_nodes']
            worker_nodes = job['worker_nodes']
            bw_per_worker = job['bw_per_worker']

            can_allocate = True

            # Check if the bandwidth can be allocated for every worker-PS pair
            for worker_node in worker_nodes:
                for ps_node in ps_nodes:
                    if not self.can_allocate_bandwidth(worker_node, ps_node, bw_per_worker):
                        can_allocate = False
                        # print(f"Job {idx+1}: Cannot allocate {bw_per_worker} Mbps between Worker {worker_node.node_id} and PS {ps_node.node_id}. Skipping allocation.")
                        break
                if not can_allocate:
                    break

            # If all bandwidth can be allocated, proceed with allocation
            if can_allocate:
                for worker_node in worker_nodes:
                    for ps_node in ps_nodes:
                        try:
                            self.topology.route_traffic(worker_node, ps_node, bw_per_worker)
                        except BandwidthAllocationError as e:
                            print(f"BandwidthAllocationError: {e}")
            else:
                # print(f"Job {idx+1}: Skipped due to insufficient bandwidth.")
                pass




    def allocate_bandwidth(self, ps_nodes, worker_nodes, bw_per_worker):
        """
        Allocate bandwidth for all jobs in the job list using a sequential allocation strategy.
        """
        # print(f"\nAllocating bandwidth for Job")
        # ps_nodes = job['ps_nodes']
        # worker_nodes = job['worker_nodes']
        # bw_per_worker = job['bw_per_worker']

        can_allocate = True

        # Check if the bandwidth can be allocated for every worker-PS pair
        for worker_node in worker_nodes:
            for ps_node in ps_nodes:
                if not self.can_allocate_bandwidth(worker_node, ps_node, bw_per_worker):
                    can_allocate = False
                    # print(f"Job: Cannot allocate {bw_per_worker} Mbps between Worker {worker_node.node_id} and PS {ps_node.node_id}. Skipping allocation.")
                    break
            if not can_allocate:
                break

        # If all bandwidth can be allocated, proceed with allocation
        if can_allocate:
            for worker_node in worker_nodes:
                for ps_node in ps_nodes:
                    try:
                        self.topology.route_traffic(worker_node, ps_node, bw_per_worker)
                    except BandwidthAllocationError as e:
                        print(f"BandwidthAllocationError: {e}")
        else:
            # print(f"Job: Skipped due to insufficient bandwidth.")
            pass

    def reset_jobs(self):
        """
        Reset all jobs and clear bandwidth allocations.
        """
        self.jobs = []
        self.topology.reset_allocations()
        # print("SequentialJobGenerator: Reset all jobs and bandwidth allocations.")


import random

# Example ranges for random values
MIN_PS_NODES = 1
MAX_PS_NODES = 2

MIN_WORKER_NODES = 2
MAX_WORKER_NODES = 5

MIN_BW_PER_WORKER = 50.0
MAX_BW_PER_WORKER = 100.0

NUM_JOBS = 1  # Number of jobs to create randomly

NUM_SPINE_SWITCHES = 5  # Number of spine switches
NUM_LEAF_SWITCHES = 10  # Number of leaf switches
NUM_NODES = 100  # Number of nodes
MAX_SPINE_CAPACITY = 500.0  # Maximum capacity for each spine switch
MAX_LEAF_CAPACITY = 300.0  # Maximum capacity for each leaf switch
MAX_NODE_BW = 100.0  # Maximum bandwidth capacity for each node

if __name__ == "__main__":
    # Initialize the topology
    topology = SpineLeafTopology()

    # Call the init_topology method to set up the topology
    topology.init_topology(num_spine_switches=NUM_SPINE_SWITCHES,
                           num_leaf_switches=NUM_LEAF_SWITCHES,
                           num_nodes=NUM_NODES,
                           max_spine_capacity=MAX_SPINE_CAPACITY,
                           max_leaf_capacity=MAX_LEAF_CAPACITY,
                           max_node_bw=MAX_NODE_BW)

    # print("\n--- Topology Setup Complete ---\n")

    # Create a few random jobs and allocate bandwidth
    for job_id in range(90):
        # Randomly generate the number of PS and worker nodes for the job
        ps_node_count = 1
        worker_node_count = 3

        # Randomly generate the bandwidth per worker
        bw_per_worker = 20

        # Randomly select node IDs for parameter servers and workers
        all_node_ids = [node.node_id for node in topology.nodes]
        ps_nodes = random.sample(all_node_ids, ps_node_count)
        ps_nodes = [0+job_id]
        available_worker_ids = [node_id for node_id in all_node_ids if node_id not in ps_nodes]
        worker_nodes = random.sample(available_worker_ids, worker_node_count)
        worker_nodes = [1+job_id,2+job_id,3+job_id]
        # print(('ktm',ps_nodes, worker_nodes, bw_per_worker))

        # print(f"\n--- Creating Job {job_id + 1}: PS Nodes {ps_nodes}, Worker Nodes {worker_nodes}, Bandwidth: {bw_per_worker:.2f} Mbps per worker ---")

        # Allocate bandwidth for the job
        topology.allocate_bandwidth(ps_nodes, worker_nodes, bw_per_worker)

        # Print the adjacency matrix for available bandwidth between nodes
        adjacency_matrix_available_bw = topology.get_adjacency_matrix()
        topology.print_adjacency_matrix(adjacency_matrix_available_bw)
        
        #     # Plot bandwidth utilization after job allocation
        topology.plot_spine_utilization()  # Call the plot method for spine utilization
        topology.plot_congestion_metric()  # Call the plot method for spine utilization
        topology.plot_bandwidth_utilization()  # Call the plot method for spine utilization
        topology.plot_utilization_metric()  # Call the plot method for spine utilization

