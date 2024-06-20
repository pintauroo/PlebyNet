"""
Topology building module
"""

import numpy as np

class topo:
    def __init__(self, func_name, max_bandwidth, min_bandwidth, num_clients, num_edges, probability=0):
        self.n = num_edges #adjacency matrix

        self.to = getattr(self, func_name)
        self.b = max_bandwidth
        self.probability = 0.4
        # self.b = np.random.uniform(min_bandwidth, max_bandwidth, size=(num_clients, num_edges)) #bandwidth matrix
        
        np.random.seed(0)
        
        if func_name == "complete_graph":
            self.adjacency_matrix = self.compute_complete_graph()
        elif func_name == "ring_graph":
            self.adjacency_matrix = self.compute_ring_graph()
        elif func_name == "star_graph":
            self.adjacency_matrix = self.compute_star_graph()
        elif func_name == "grid_graph":
            self.adjacency_matrix = self.compute_grid_graph()
        elif func_name == "linear_topology":
            self.adjacency_matrix = self.compute_linear_topology()
        elif func_name == "probability_graph":
            self.adjacency_matrix = self.compute_probabilistic_graph()
        else:
            raise ValueError("Invalid topology function name")
        
        self.adjacency_matrix *= 10
        
        np.savetxt('topology.csv', self.adjacency_matrix, delimiter=',', fmt='%d')

    def call_func(self):
        return self.to()
    
    def compute_linear_topology(self):
        """
        This function returns the adjacency matrix for a linear topology of n nodes.
        """
        # Create an empty adjacency matrix of size n x n
        adjacency_matrix = np.zeros((self.n, self.n))
        
        # Add edges to the adjacency matrix
        for i in range(self.n):
            if i == 0:
                # Connect node 0 to node 1
                adjacency_matrix[i][i+1] = 1
            elif i == self.n-1:
                # Connect node n-1 to node n-2
                adjacency_matrix[i][i-1] = 1
            else:
                # Connect node i to nodes i-1 and i+1
                adjacency_matrix[i][i-1] = 1
                adjacency_matrix[i][i+1] = 1
        
        return adjacency_matrix
    
    def linear_topology(self):
        return self.adjacency_matrix

    def compute_complete_graph(self):
        adjacency_matrix = np.ones((self.n, self.n)) - np.eye(self.n)
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
            for j in range(i + 1, self.n):  # Ensure j > i to avoid redundant assignments
                value = np.random.choice([0, 1], p=[1-self.probability, self.probability])
                adjacency_matrix[i][j] = value
                adjacency_matrix[j][i] = value
        return adjacency_matrix
    
    def probability_graph(self):
        return self.adjacency_matrix
    
    def detach_node(self, node):
        self.adjacency_matrix[node, :] = 0
        self.adjacency_matrix[:, node] = 0

