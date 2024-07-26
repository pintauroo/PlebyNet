"""
Topology building module
"""

from matplotlib import pyplot as plt
import numpy as np

class Topology:
    def __init__(self, func_name, max_bandwidth, num_nodes, probability=0.4):
        self.n = num_nodes 
        self.to = getattr(self, func_name)
        self.b = max_bandwidth
        self.probability = probability
        
        np.random.seed(0)
        
        if func_name == "compute_complete_graph":
            self.adjacency_matrix = self.compute_complete_graph()
        elif func_name == "compute_ring_graph":
            self.adjacency_matrix = self.compute_ring_graph()
        elif func_name == "compute_star_graph":
            self.adjacency_matrix = self.compute_star_graph()
        elif func_name == "compute_grid_graph":
            self.adjacency_matrix = self.compute_grid_graph()
        elif func_name == "compute_linear_topology":
            self.adjacency_matrix = self.compute_linear_topology()
        elif func_name == "compute_probabilistic_graph":
            self.adjacency_matrix = self.compute_probabilistic_graph()
        else:
            raise ValueError("Invalid topology function name")
        
        self.bandwidth_matrix = self.adjacency_matrix.copy() 
        self.bandwidth_matrix = self.adjacency_matrix * self.b
        self.bandwidth_matrix_updated = self.bandwidth_matrix.copy()
        
        np.savetxt('topology.csv', self.adjacency_matrix, delimiter=',', fmt='%d')

    def call_func(self):
        return self.to()


    # Getter for the local matrix
    def get_adjacency_matrix(self):
        return self.adjacency_matrix.astype(int).tolist()
    def get_updated_bw_matrix(self):
        return self.bandwidth_matrix_updated.astype(int).tolist()
    def get_initial_bw_matrix(self):
        return self.bandwidth_matrix.astype(int).tolist()
    def get_total_initial_bw(self):
        initial_total_bandwidth = 0
        for i in range(self.n):
            initial_total_bandwidth += max(self.bandwidth_matrix[i])
        return initial_total_bandwidth
    
    def get_total_remaining_bw(self):
        updated_total_bandwidth = 0
        for i in range(self.n):
            updated_total_bandwidth += max(self.bandwidth_matrix_updated[i])
        return updated_total_bandwidth
    
    def get_total_allocated_bw(self):
        return self.get_total_initial_bw() - self.get_total_remaining_bw()
    
    def get_total_percentage_bw_used(self):
        return (self.get_total_remaining_bw() / self.get_total_initial_bw() ) * 100
    

    
    
        


    # Function to increase the value for the local copy of the topology given two indexes
    def increase_value(self, index1, index2, value=1):
        self.bandwidth_matrix_updated[index1, index2] += value
        self.bandwidth_matrix_updated[index2, index1] += value  # Assuming undirected graph

    # Function to decrease the value for the local copy of the topology given two indexes
    def decrease_value(self, index1, index2, value):
        self.bandwidth_matrix_updated[index1, index2] = max(0, self.bandwidth_matrix_updated[index1, index2] - value)
        self.bandwidth_matrix_updated[index2, index1] = max(0, self.bandwidth_matrix_updated[index2, index1] - value)  # Assuming undirected graph


    # def allocate_bandwidth(self, node1, node2, bandwidth):
    #     # Allocate the bandwidth between node1 and node2
    #     if (self.bandwidth_matrix_updated[node1, node2] >= bandwidth and 
    #         self.bandwidth_matrix_updated[node2, node1] >= bandwidth):

    #         self.bandwidth_matrix_updated[node1, node2] -= bandwidth
    #         self.bandwidth_matrix_updated[node2, node1] -= bandwidth
    #         # Adjust the bandwidth for all other connections of node1 and node2
    #         for i in range(self.n):
    #             if i != node2:
    #                 self.bandwidth_matrix_updated[node1, i] = max(0, self.bandwidth_matrix_updated[node1, i] - bandwidth)
    #                 self.bandwidth_matrix_updated[i, node1] = max(0, self.bandwidth_matrix_updated[i, node1] - bandwidth)
    #             if i != node1:
    #                 self.bandwidth_matrix_updated[node2, i] = max(0, self.bandwidth_matrix_updated[node2, i] - bandwidth)
    #                 self.bandwidth_matrix_updated[i, node2] = max(0, self.bandwidth_matrix_updated[i, node2] - bandwidth)
    #         print(f"Allocated {bandwidth} bandwidth between node {node1} and node {node2}")
    #         return True
    #     else:
    #         print(f"Insufficient bandwidth to allocate {bandwidth} between node {node1} and node {node2}")
    #         return False
    
    def allocate_bandwidth(self, node1, node2, bandwidth):
        # Allocate the bandwidth between node1 and node2
        if (self.bandwidth_matrix_updated[node1][node2] >= bandwidth and 
            self.bandwidth_matrix_updated[node2][node1] >= bandwidth):

            self.bandwidth_matrix_updated[node1][node2] -= bandwidth
            self.bandwidth_matrix_updated[node2][node1] -= bandwidth

            # Adjust the bandwidth for all other connections of node1 and node2
            for i in range(self.n):
                if i != node1 and i != node2:
                    self.bandwidth_matrix_updated[node1][i] = max(0, self.bandwidth_matrix_updated[node1][i] - bandwidth)
                    self.bandwidth_matrix_updated[i][node1] = max(0, self.bandwidth_matrix_updated[i][node1] - bandwidth)
                    self.bandwidth_matrix_updated[node2][i] = max(0, self.bandwidth_matrix_updated[node2][i] - bandwidth)
                    self.bandwidth_matrix_updated[i][node2] = max(0, self.bandwidth_matrix_updated[i][node2] - bandwidth)
            
            print(f"Allocated {bandwidth} bandwidth between node {node1} and node {node2}")
            return True
        else:
            print(f"Insufficient bandwidth to allocate {bandwidth} between node {node1} and node {node2}")
            return False

    def compute_linear_topology(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i == 0:
                adjacency_matrix[i][i+1] = 1
            elif i == self.n-1:
                adjacency_matrix[i][i-1] = 1
            else:
                adjacency_matrix[i][i-1] = 1
                adjacency_matrix[i][i+1] = 1
        return adjacency_matrix

    def linear_topology(self):
        return self.adjacency_matrix

    def compute_complete_graph(self):
        adjacency_matrix = np.ones((self.n, self.n), dtype=int) - np.eye(self.n, dtype=int)
        return adjacency_matrix
        
    def complete_graph(self):
        return self.adjacency_matrix

    def compute_ring_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            adjacency_matrix[i][(i-1)%self.n] = 1
            adjacency_matrix[i][(i+1)%self.n] = 1
        return adjacency_matrix

    def ring_graph(self):
        return self.adjacency_matrix

    def compute_star_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        adjacency_matrix[0,:] = 1
        adjacency_matrix[:,0] = 1
        adjacency_matrix[0,0] = 0
        return adjacency_matrix

    def star_graph(self):
        return self.adjacency_matrix

    def compute_grid_graph(self):
        adjacency_matrix = np.zeros((self.n*self.n, self.n*self.n))
        for i in range(self.n):
            for j in range(self.n):
                node = i*self.n + j
                if i > 0:
                    adjacency_matrix[node][node-self.n] = 1
                if i < self.n-1:
                    adjacency_matrix[node][node+self.n] = 1
                if j > 0:
                    adjacency_matrix[node][node-1] = 1
                if j < self.n-1:
                    adjacency_matrix[node][node+1] = 1
        return adjacency_matrix

    def grid_graph(self):
        return self.adjacency_matrix

    def compute_probabilistic_graph(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                value = np.random.choice([0, 1], p=[1-self.probability, self.probability])
                adjacency_matrix[i][j] = value
                adjacency_matrix[j][i] = value
        return adjacency_matrix

    def probability_graph(self):
        return self.adjacency_matrix

    def detach_node(self, node):
        self.adjacency_matrix[node, :] = 0
        self.adjacency_matrix[:, node] = 0

    # Method to calculate the total occupied bandwidth
    def calculate_occupied_bandwidth(self):
        initial_total_bw = np.sum(self.bandwidth_matrix)
        updated_total_bw = np.sum(self.bandwidth_matrix_updated)
        occupied_bw = initial_total_bw - updated_total_bw
        return occupied_bw

    # Method to plot the initial and updated bandwidth matrices as heatmaps
    def plot_bandwidth_matrices(self, probability, max_bw):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Initial bandwidth matrix heatmap
        im1 = axes[0].imshow(self.bandwidth_matrix, cmap='viridis', aspect='auto')
        axes[0].set_title('Initial Bandwidth Matrix')
        fig.colorbar(im1, ax=axes[0])

        # Updated bandwidth matrix heatmap
        im2 = axes[1].imshow(self.bandwidth_matrix_updated, cmap='viridis', aspect='auto')
        axes[1].set_title('Updated Bandwidth Matrix')
        fig.colorbar(im2, ax=axes[1])

        plt.savefig('bandwidth_matrices'+str(probability)+''+str(max_bw)+''+'.png')

    # Method to plot the total occupied bandwidth over time
    def plot_occupied_bandwidth(self, occupied_bandwidths):
        plt.figure(figsize=(8, 6))
        plt.plot(occupied_bandwidths, marker='o', linestyle='-')
        plt.title('Total Occupied Bandwidth Over Time')
        plt.xlabel('Allocation Step')
        plt.ylabel('Occupied Bandwidth')
        plt.grid(True)
        plt.savefig('plot_occupied_bandwidth.png')

# Example usage:
# topo = Topology('compute_linear_topology', 10, 1, 5, 5)
# print(topo.get_local_matrix())
# topo.increase_value(0, 1, 5)
# print(topo.get_local_matrix())
