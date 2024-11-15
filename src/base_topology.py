import networkx as nx
from typing import Dict, Any

class BaseTopology:
    def __init__(self):
        """
        Initializes the BaseTopology with an empty graph.
        """
        self.graph = nx.Graph()

    def create_topology(self) -> None:
        """
        Method to create the specific topology. 
        Should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def add_node(self, node_id: str, **attributes: Dict[str, Any]) -> None:
        """
        Adds a node to the topology.

        Args:
            node_id (str): The unique identifier for the node.
            attributes (dict): Optional attributes associated with the node.
        """
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, node1: str, node2: str, **attributes: Dict[str, Any]) -> None:
        """
        Adds an edge between two nodes with specified attributes (e.g., bandwidth).

        Args:
            node1 (str): The first node in the edge.
            node2 (str): The second node in the edge.
            attributes (dict): Optional attributes associated with the edge.
        """
        self.graph.add_edge(node1, node2, **attributes)

    def get_graph(self) -> nx.Graph:
        """
        Returns the graph object representing the topology.

        Returns:
            nx.Graph: The NetworkX graph object.
        """
        return self.graph

class CustomTopology(BaseTopology):
    def create_topology(self):
        self.add_node('A', type='switch')
        self.add_node('B', type='host')
        self.add_edge('A', 'B', bandwidth=100)

topology = CustomTopology()
topology.create_topology()
graph = topology.get_graph()
