from csv import DictWriter
import os
import random
import threading
import sys
# import pygraphviz as pgv
import sys
from collections import deque
from enum import Enum
from src.base_topology import BaseTopology
import networkx as nx

# class syntax
network_aware = True

class TopologyType(Enum):
    RING = 1
    FAT_TREE = 2


def dijkstra(adj_matrix, start_node, end_node):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes  # Track visited nodes
    # Initialize distances with a large value
    distances = [sys.maxsize] * num_nodes
    distances[start_node] = 0  # Set distance of the start node to 0

    # Find the shortest path for all nodes
    for _ in range(num_nodes):
        min_distance = sys.maxsize
        min_node = -1

        # Find the node with the smallest distance from the set of unvisited nodes
        for node in range(num_nodes):
            if not visited[node] and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        visited[min_node] = True

        # Update distances for the adjacent nodes
        for node in range(num_nodes):
            if (
                not visited[node]
                and adj_matrix[min_node][node] == 1
                and distances[min_node] + 1 < distances[node]
            ):
                distances[node] = distances[min_node] + 1

    # Backtrack to find the shortest path
    if distances[end_node] == sys.maxsize:
        # No path exists
        return None, None

    path = deque()
    current_node = end_node
    path.appendleft(current_node)

    while current_node != start_node:
        for node in range(num_nodes):
            if (
                adj_matrix[node][current_node] == 1
                and distances[node] + 1 == distances[current_node]
            ):
                current_node = node
                path.appendleft(current_node)
                break

    return list(path), distances[end_node]


class NetworkTopology:
    class Edge:
        def __init__(self, id, bw):
            self.__id = id
            self.__bw = bw
            self.__initial_bw = bw

        def get_id(self):
            return str(self.__id)

        def get_bw(self):
            return self.__bw
        
        def get_initial(self):
            return self.__initial_bw

        def consume_bw(self, bw):
            self.__bw -= bw

        def release_bw(self, bw):
            self.__bw += bw

        def get_resource_usage(self):
            return ((self.__initial_bw - self.__bw)/self.__initial_bw)*100

        def __str__(self) -> str:
            return "Edge_" + str(self.__id)

    def __init__(self, n_nodes, min_bw, max_bw, group_number=3, seed=None, topology_type=TopologyType.RING):
        if seed is not None:
            random.seed(seed)

        self.__group_number = group_number
        self.__n_nodes = n_nodes
        self.__min_bw = min_bw
        self.__max_bw = max_bw
        self.__lock = threading.Lock()
        self.__topology_type = topology_type
        self.__node_operations = {}
        self.__client_operations = {}

        self.__generate_topology()

    def __generate_topology(self):
        if self.__n_nodes < self.__group_number:
            print(
                "Number of nodes in the network topology must be >= than the number of groups. Exiting...")
            # sys.exit(1)

        if self.__group_number < 2:
            print(
                "Number of groups in the network topology must be >= than 2. Exiting...")
            sys.exit(1)

        self.__edges = {}

        if self.__topology_type == TopologyType.RING:
            self.__generate_ring_topology()
        elif self.__topology_type == TopologyType.FAT_TREE:
            self.__generate_fat_tree_topolgy()
        else:
            print("Invalid topology type: ", self.__topology_type)
            sys.exit(1)

        self.__path = [[] for _ in range(self.__n_nodes+1)]
        for i in range(self.__n_nodes+1):
            for j in range(self.__n_nodes+1):
                self.__path[i].append([])

        for i in range(self.__n_nodes):
            for j in range(self.__n_nodes-(self.__n_nodes-i)):
                if i != j:
                    shortest_path, _ = dijkstra(self.__connected, i, j)
                    # print(f"From {i} to {j}: ", end='')
                    for k in range(len(shortest_path)-1):
                        # print(f"Edge: {self.__edge_id[shortest_path[k]][shortest_path[k+1]]} ", end='')
                        self.__path[i][j].append(
                            self.__edge_id[shortest_path[k]][shortest_path[k+1]])
                        self.__path[j][i].append(
                            self.__edge_id[shortest_path[k]][shortest_path[k+1]])
                    # print()

        for j in range(self.__n_nodes):
            shortest_path, _ = dijkstra(
                self.__connected, len(self.__connected)-1, j)
            # print(f"From Client to {j}: ", end='')
            for k in range(len(shortest_path)-1):
                # print(f"Edge: {self.__edge_id[shortest_path[k]][shortest_path[k+1]]} ", end='')
                self.__path[len(
                    self.__path)-1][j].append(self.__edge_id[shortest_path[k]][shortest_path[k+1]])
                self.__path[j][len(
                    self.__path)-1].append(self.__edge_id[shortest_path[k]][shortest_path[k+1]])
            # print()

        #self.__export_as_dot()

    def __generate_fat_tree_topolgy(self):
        self.__edge_id = [[]
                          for _ in range(self.__n_nodes+self.__group_number+3)]
        for i in range(self.__n_nodes+self.__group_number+3):
            for j in range(self.__n_nodes+self.__group_number+3):
                self.__edge_id[i].append([])

        self.__connected = [[]
                            for _ in range(self.__n_nodes+self.__group_number+3)]
        for i in range(self.__n_nodes+self.__group_number+3):
            for j in range(self.__n_nodes+self.__group_number+3):
                self.__connected[i].append([])
                self.__connected[i][j] = 0

        node_group = []
        id = 0
        for i in range(self.__n_nodes):
            node_group.append(id)
            id = (id+1) % self.__group_number

        # edge dai nodi agli switch
        index = 0
        self.__direct_edge_id = {}
        for i in range(self.__n_nodes):
            e = NetworkTopology.Edge(
                index, random.uniform(self.__min_bw, self.__max_bw))
            self.__edges[e.get_id()] = e
            self.__direct_edge_id[i] = e.get_id()
            index += 1
            self.__edge_id[i][self.__n_nodes + node_group[i]] = e.get_id()
            self.__edge_id[self.__n_nodes + node_group[i]][i] = e.get_id()
            self.__connected[i][self.__n_nodes + node_group[i]] = 1
            self.__connected[self.__n_nodes + node_group[i]][i] = 1

        # edge tra gli switch
        id = 0
        for i in range(self.__group_number):
            next_id = self.__n_nodes + self.__group_number + id
            id = (id+1) % 2

            e = NetworkTopology.Edge(
                index, random.uniform(self.__min_bw, self.__max_bw))
            self.__edges[e.get_id()] = e
            index += 1

            self.__edge_id[self.__n_nodes+i][next_id] = e.get_id()
            self.__edge_id[next_id][self.__n_nodes+i] = e.get_id()
            self.__connected[self.__n_nodes+i][next_id] = 1
            self.__connected[next_id][self.__n_nodes+i] = 1

        # edge tra i due switch di backbone
        e = NetworkTopology.Edge(
            index, random.uniform(self.__min_bw, self.__max_bw))
        self.__edges[e.get_id()] = e
        index += 1

        self.__edge_id[self.__n_nodes+self.__group_number][self.__n_nodes +
                                                           self.__group_number+1] = e.get_id()
        self.__edge_id[self.__n_nodes+self.__group_number +
                       1][self.__n_nodes+self.__group_number] = e.get_id()
        self.__connected[self.__n_nodes +
                         self.__group_number][self.__n_nodes+self.__group_number+1] = 1
        self.__connected[self.__n_nodes+self.__group_number +
                         1][self.__n_nodes+self.__group_number] = 1

        # client
        e = NetworkTopology.Edge(index, float('inf'))
        self.__edges[e.get_id()] = e
        index += 1

        self.__edge_id[self.__n_nodes+self.__group_number][self.__n_nodes +
                                                           self.__group_number+2] = e.get_id()
        self.__edge_id[self.__n_nodes+self.__group_number +
                       2][self.__n_nodes+self.__group_number] = e.get_id()
        self.__connected[self.__n_nodes +
                         self.__group_number][self.__n_nodes+self.__group_number+2] = 1
        self.__connected[self.__n_nodes+self.__group_number +
                         2][self.__n_nodes+self.__group_number] = 1

    def __generate_ring_topology(self):
        self.__edge_id = [[]
                          for _ in range(self.__n_nodes+self.__group_number+1)]
        for i in range(self.__n_nodes+self.__group_number+1):
            for j in range(self.__n_nodes+self.__group_number+1):
                self.__edge_id[i].append([])

        self.__connected = [[]
                            for _ in range(self.__n_nodes+self.__group_number+1)]
        for i in range(self.__n_nodes+self.__group_number+1):
            for j in range(self.__n_nodes+self.__group_number+1):
                self.__connected[i].append([])
                self.__connected[i][j] = 0

        node_group = []
        id = 0
        for i in range(self.__n_nodes):
            node_group.append(id)
            id = (id+1) % self.__group_number

        self.__direct_edge_id = {}
        # edge dai nodi agli switch
        index = 0
        for i in range(self.__n_nodes):
            e = NetworkTopology.Edge(
                index, random.uniform(self.__min_bw, self.__max_bw))
            self.__edges[e.get_id()] = e
            self.__direct_edge_id[i] = e.get_id()
            index += 1
            self.__edge_id[i][self.__n_nodes + node_group[i]] = e.get_id()
            self.__edge_id[self.__n_nodes + node_group[i]][i] = e.get_id()
            self.__connected[i][self.__n_nodes + node_group[i]] = 1
            self.__connected[self.__n_nodes + node_group[i]][i] = 1

        # edge tra gli switch
        for i in range(self.__group_number):
            next_id = self.__n_nodes + (i+1) % self.__group_number
            prev_id = self.__n_nodes + self.__group_number - \
                1 if i == 0 else self.__n_nodes + i-1
            e1 = NetworkTopology.Edge(
                index, random.uniform(self.__min_bw, self.__max_bw))
            self.__edges[e1.get_id()] = e1
            index += 1
            e2 = NetworkTopology.Edge(
                index, random.uniform(self.__min_bw, self.__max_bw))
            self.__edges[e2.get_id()] = e2
            index += 1
            self.__edge_id[self.__n_nodes+i][prev_id] = e1.get_id()
            self.__edge_id[self.__n_nodes+i][next_id] = e2.get_id()
            self.__edge_id[prev_id][self.__n_nodes+i] = e1.get_id()
            self.__edge_id[next_id][self.__n_nodes+i] = e2.get_id()
            self.__connected[self.__n_nodes+i][prev_id] = 1
            self.__connected[self.__n_nodes+i][next_id] = 1
            self.__connected[prev_id][self.__n_nodes+i] = 1
            self.__connected[next_id][self.__n_nodes+i] = 1

        # client
        e = NetworkTopology.Edge(index, float('inf'))
        self.__edges[e.get_id()] = e
        index += 1

        self.__edge_id[self.__n_nodes+self.__group_number][self.__n_nodes +
                                                           self.__group_number-1] = e.get_id()
        self.__edge_id[self.__n_nodes+self.__group_number -
                       1][self.__n_nodes+self.__group_number] = e.get_id()
        self.__connected[self.__n_nodes +
                         self.__group_number][self.__n_nodes+self.__group_number-1] = 1
        self.__connected[self.__n_nodes+self.__group_number -
                         1][self.__n_nodes+self.__group_number] = 1

    def get_available_bandwidth_between_nodes(self, id1, id2):
        if id1 == float('-inf') and id2 == float('-inf'):
            print("Something wrong happened. Trying to get the available bw between non existing nodes. Exiting...")
            sys.exit(1)

        if not network_aware:
            return self.get_node_direct_link_bw(id1)
        
        if id1 == id2:
            return float('inf')

        with self.__lock:
            # if one of the node is inf, return the minimum bw value from the other 
            # Note: could be changed with the average
            if id1 == float('-inf') or id2 == float('-inf'):
                index = id1 if id1 != float('-inf') else id2
                min_bw = float('inf')
                for i in range(self.__n_nodes):
                    if i != index:
                        edges = self.__path[index][i]
                        for e_id in edges:
                            if self.__edges[e_id].get_bw() < min_bw:
                                min_bw = self.__edges[e_id].get_bw()
            else:
                edges = self.__path[id1][id2]
                min_bw = float('inf')
                for e_id in edges:
                    if self.__edges[e_id].get_bw() < min_bw:
                        min_bw = self.__edges[e_id].get_bw()
            return min_bw

    def consume_bandwidth_between_nodes(self, id1, id2, bw, job_id):
        # print(f"Consuming bw between {id1} and {id2} -- Job {job_id}", flush=True)

        with self.__lock:
            if job_id not in self.__node_operations:
                self.__node_operations[job_id] = {}
            if id1 == id2:
                return True                
            edges = self.__path[id1][id2]
            
            if network_aware:
                for e_id in edges:
                    if self.__edges[e_id].get_bw() < bw:
                        return False
            
            for e_id in edges:
                self.__edges[e_id].consume_bw(bw)
            
            key = str(min(id1, id2)) + "_" + str(max(id1, id2))
            if key not in self.__node_operations[job_id]:
                self.__node_operations[job_id][key] = 1
            else:
                self.__node_operations[job_id][key] += 1
            return True

    def release_bandwidth_between_nodes(self, id1, id2, bw, job_id):
        # print(f"Releasing bw between {id1} and {id2} -- Job {job_id}", flush=True)
        if id1 == id2:
            return

        with self.__lock:
            key = str(min(id1, id2)) + "_" + str(max(id1, id2))
            self.__node_operations[job_id][key] -= 1
            edges = self.__path[id1][id2]
            for e_id in edges:
                self.__edges[e_id].release_bw(bw)

    def get_available_bandwidth_with_client(self, id1):
        if not network_aware:
            return self.get_node_direct_link_bw(id1)

        with self.__lock:
            edges = self.__path[len(self.__path)-1][id1]
            min_bw = float('inf')
            for e_id in edges:
                if self.__edges[e_id].get_bw() < min_bw:
                    min_bw = self.__edges[e_id].get_bw()
            return min_bw

    def consume_bandwidth_node_and_client(self, id1, bw, job_id):
        # print(f"Consuming bw between {id1} and Client -- Job {job_id}", flush=True)
        with self.__lock:
            if job_id not in self.__client_operations:
                self.__client_operations[job_id] = {}
                
            edges = self.__path[len(self.__path)-1][id1]

            if network_aware:
                for e_id in edges:
                    if self.__edges[e_id].get_bw() < bw:
                        return False
            for e_id in edges:
                self.__edges[e_id].consume_bw(bw)
                
            if str(id1) not in self.__client_operations[job_id]:
                self.__client_operations[job_id][str(id1)] = 1
            else:
                self.__client_operations[job_id][str(id1)] += 1
            return True

    def release_bandwidth_node_and_client(self, id1, bw, job_id):
        # print(f"Releasing bw between {id1} and Client -- Job {job_id}", flush=True)
        with self.__lock:
            self.__client_operations[job_id][str(id1)] -= 1
            edges = self.__path[len(self.__path)-1][id1]
            for e_id in edges:
                self.__edges[e_id].release_bw(bw)

    def get_node_direct_link_bw(self, id):
        with self.__lock:
            return self.__edges[self.__direct_edge_id[id]].get_bw()
        
    def check_network_consistency(self, bids):
        print("Performing consistency check on network topology...")
        
        for key, bid in bids.items():
            expected_node_allocation = 0
            client_allocations = 0
            val = bid[0]
            for b in bid:
                if val != b:
                    val = b
                    expected_node_allocation += 1 
            
            if key in self.__node_operations:
                for key2 in self.__node_operations[key]:
                    if key in self.__node_operations:
                        expected_node_allocation -= self.__node_operations[key][key2]
                
            if expected_node_allocation != 0:
                print("Too many bandwidth reservation requests between nodes")
                print(f"There's a problem with job {key}")
                print(self.__node_operations[key])
                sys.exit(1)
            if key in self.__client_operations:    
                for key2 in self.__client_operations[key]:
                    if key in self.__client_operations:
                        client_allocations += self.__client_operations[key][key2]
                
            if client_allocations != 1:
                print("Too many bandwidth reservation requests between node and client")
                print(f"There's a problem with job {key}")
                print(self.__client_operations[key])
                sys.exit(1)
            
        print("The network topology is consistent with the final allocation scheme")
        return 
                    
    def __print_topology(self):
        for i in range(self.__n_nodes):
            print(f"Node {i}: ", end="")
            for j in range(self.__n_nodes):
                if i != j:
                    print(f"--> Node {j} (", end="")
                    for e in self.adj[i][j]:
                        print(e, end=" ")
                    print(") ", end="")
            print()
            
    def dump_to_file(self, filename, alpha):
        ret_dict = {}
        ret_dict["alpha"] = alpha
        
        for key in self.__edges:
            if self.__edges[key].get_initial() != float('inf'): 
                ret_dict["Edge_" + str(key) + "_initial"] = round(self.__edges[key].get_initial())  
                ret_dict["Edge_" + str(key) + "_final"] = round(self.__edges[key].get_bw())
                ret_dict["Edge_" + str(key) + "_usage"] = round(self.__edges[key].get_resource_usage())   
            
        self.__write_data(ret_dict.keys(), ret_dict, filename)     
            
    def __write_data(self, field_names, dictionary, filename):
        filename = str(filename)+'_bw_usage.csv'

        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as f: 
            writer = DictWriter(f, fieldnames=field_names)
        
            if not file_exists:
                writer.writeheader() 
        
            writer.writerow(dictionary)

    def __export_as_dot(self):
        if self.__topology_type == TopologyType.RING:
            G = pgv.AGraph(strict=False, directed=False)
            for i in range(self.__n_nodes+self.__group_number):
                for j in range(self.__n_nodes+self.__group_number - (self.__n_nodes+self.__group_number-i)):
                    if self.__connected[i][j] == 1:
                        if i < self.__n_nodes and j >= self.__n_nodes:
                            G.add_edge("Node " + str(i), "Sw " +
                                       str(j), label=self.__edge_id[i][j])
                        if i < self.__n_nodes and j < self.__n_nodes:
                            G.add_edge("Node " + str(i), "Node " +
                                       str(j), label=self.__edge_id[i][j])
                        if i >= self.__n_nodes and j >= self.__n_nodes:
                            G.add_edge("Sw " + str(i), "Sw " +
                                       str(j), label=self.__edge_id[i][j])
                        if i >= self.__n_nodes and j < self.__n_nodes:
                            G.add_edge("Sw " + str(i), "Node " +
                                       str(j), label=self.__edge_id[i][j])
            if self.__connected[self.__n_nodes+self.__group_number][self.__n_nodes+self.__group_number-1] == 1:
                G.add_edge("Client", "Sw " + str(self.__n_nodes+self.__group_number-1),
                           label=self.__edge_id[self.__n_nodes+self.__group_number-1][self.__n_nodes+self.__group_number])
            G.write("file.dot")
        elif self.__topology_type == TopologyType.FAT_TREE:
            G = pgv.AGraph(strict=False, directed=False)
            for i in range(self.__n_nodes+self.__group_number+2):
                for j in range(self.__n_nodes+self.__group_number+2 - (self.__n_nodes+self.__group_number-i+2)):
                    if self.__connected[i][j] == 1:
                        if i < self.__n_nodes and j >= self.__n_nodes:
                            G.add_edge("Node " + str(i), "Sw " +
                                       str(j), label=self.__edge_id[i][j])
                        if i < self.__n_nodes and j < self.__n_nodes:
                            G.add_edge("Node " + str(i), "Node " +
                                       str(j), label=self.__edge_id[i][j])
                        if i >= self.__n_nodes and j >= self.__n_nodes:
                            G.add_edge("Sw " + str(i), "Sw " +
                                       str(j), label=self.__edge_id[i][j])
                        if i >= self.__n_nodes and j < self.__n_nodes:
                            G.add_edge("Sw " + str(i), "Node " +
                                       str(j), label=self.__edge_id[i][j])
            if self.__connected[self.__n_nodes+self.__group_number][self.__n_nodes+self.__group_number+2] == 1:
                G.add_edge("Client", "Sw " + str(self.__n_nodes+self.__group_number),
                           label=self.__edge_id[self.__n_nodes+self.__group_number][self.__n_nodes+self.__group_number+2])
            G.write("file.dot")

class FatTreeTopology(BaseTopology):
    def __init__(self, num_core_switches, num_agg_switches, num_edge_switches, num_hosts):
        super().__init__()
        self.num_core_switches = num_core_switches
        self.num_agg_switches = num_agg_switches
        self.num_edge_switches = num_edge_switches
        self.num_hosts = num_hosts

    def create_topology(self):
        # Create the graph
        self.graph = nx.Graph()

        # Add core switches
        core_switches = [f"core_{i}" for i in range(self.num_core_switches)]
        self.graph.add_nodes_from(core_switches)

        # Add aggregation switches
        agg_switches = [f"agg_{i}" for i in range(self.num_agg_switches)]
        self.graph.add_nodes_from(agg_switches)

        # Add edge switches
        edge_switches = [f"edge_{i}" for i in range(self.num_edge_switches)]
        self.graph.add_nodes_from(edge_switches)

        # Add hosts
        hosts = [f"host_{i}" for i in range(self.num_hosts)]
        self.graph.add_nodes_from(hosts)

        # Create connections (this is simplified and assumes a balanced Fat-Tree)
        # Core switches connect to aggregation switches
        for core in core_switches:
            for agg in agg_switches:
                self.graph.add_edge(core, agg, bandwidth=10)

        # Aggregation switches connect to edge switches
        for agg in agg_switches:
            for edge in edge_switches:
                self.graph.add_edge(agg, edge, bandwidth=10)

        # Edge switches connect to hosts
        for edge in edge_switches:
            for host in hosts:
                self.graph.add_edge(edge, host, bandwidth=1)
        return self.graph
    


