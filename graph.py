import copy
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Union


from node import Node
from part import Part

TEMPORARY_VISUALIZATION_PATH = "temporary_graph_plot.png"


class Graph:
    """
    A class to represent graphs. A Graph is composed of nodes and edges between the nodes.
    Specifically, these are *undirected*, *unweighted*, *non-cyclic* and *connected* graphs.
    """

    def __init__(self, construction_id: int = None):
        self.__construction_id: int = construction_id  # represents unix timestamp of creation date
        self.__nodes: Set[Node] = set()
        self.__edges: Dict[Node, List[Node]] = {}
        self.__node_counter: int = 0  # internal node id counter
        self.__is_connected: bool = None  # determines if the graph is connected
        self.__contains_cycle: bool = None   # determines if the graph contains non-trivial cycles
        self.__is_bidirectional: bool = None  # determines if all edges are bidirectional

    def __eq__(self, other) -> bool:
        """ Specifies equality of two graph instances. """
        if other is None:
            return False
        if not isinstance(other, Graph):
            raise TypeError(f'Can not compare different types ({type(self)} and {type(other)})')

        # Equality is defined based on the *parts*, the node id is not relevant
        # compare number of nodes
        self_nodes_sorted = sorted(self.get_edges().keys())
        other_nodes_sorted = sorted(other.get_edges().keys())
        if len(self_nodes_sorted) != len(other_nodes_sorted):
            return False
        for node_idx in range(len(self_nodes_sorted)):
            # check that parts of sorted nodes in both instances are equivalent
            self_node = self_nodes_sorted[node_idx]
            other_node = other_nodes_sorted[node_idx]
            if not self_node.get_part().equivalent(other_node.get_part()):
                return False
            # check that edges are equivalent
            self_node_neighbors = self.get_edges()[self_node]
            other_node_neighbors = other.get_edges()[other_node]
            if len(self_node_neighbors) != len(other_node_neighbors):
                return False
            for neighbor_idx in range(len(self_node_neighbors)):
                if not self_node_neighbors[neighbor_idx].get_part() \
                        .equivalent(other_node_neighbors[neighbor_idx].get_part()):
                    return False
        return True

    def __hash__(self) -> int:
        """ Defines hash of a graph. """
        hash_value = hash("Graph")
        for k, v in self.get_edges().items():
            pair = (k, tuple(v))
            hash_value = hash(hash_value + hash(pair))
        return hash_value

    def __get_node_for_part(self, part: Part) -> Node:
        """
        Returns a node of the graph for the given part. If the part is already known in the graph, the
        corresponding node is returned, else a new node is created.
        :param part: part
        :return: corresponding node for the given part
        """
        if part not in self.get_parts():
            # create new node for part
            node = Node(self.__node_counter, part)
            self.__node_counter += 1
        else:
            node = [node for node in self.get_nodes() if node.get_part() is part][0]

        return node

    def add_node(self, node):
        """ Adds a node to the internal set of nodes. """
        self.__nodes.add(node)

    def add_undirected_edge(self, part1: Part, part2: Part):
        """
        Adds an undirected edge between part1 and part2. Therefor, the parts are transformed to nodes.
        This is equivalent to adding two directed edges, one from part1 to part2 and the second from
        part2 to part1.
        :param part1: one of the parts for the undirected edge
        :param part2: second part for the undirected edge
        """
        self.__add_edge(part1, part2)
        self.__add_edge(part2, part1)

    def add_node_without_edge(self, part: Part):
        node = Node(self.__node_counter, part)
        self.__node_counter += 1
        self.add_node(node)
        self.__edges[node] = []

    def __add_edge(self, source: Part, sink: Part):
        """
        Adds an directed edge from source to sink. Therefor, the parts are transformed to nodes.
        :param source: start node of the directed edge
        :param sink: end node of the directed edge
        """
        # do not allow self-loops of a node on itself
        if source == sink:
            return

        # adding edges influences if the graph is connected and cyclic
        self.__is_connected = None
        self.__contains_cycle = None

        source_node = self.__get_node_for_part(source)
        self.add_node(source_node)
        sink_node = self.__get_node_for_part(sink)
        self.add_node(sink_node)

        # check if source node has already outgoing edges
        if source_node not in self.get_edges().keys():
            self.__edges[source_node] = [sink_node]  # values of dict need to be arrays
        else:
            connected_nodes = self.get_edges().get(source_node)
            # check if source and sink are already connected (to ignore duplicate connection)
            if sink_node not in connected_nodes:
                self.__edges[source_node] = sorted(connected_nodes + [sink_node])

    def get_node(self, node_id: int):
        """ Returns the corresponding node for a given node id. """
        matching_nodes = [node for node in self.get_nodes() if node.get_id() is node_id]
        if not matching_nodes:
            raise AttributeError('Given node id not found.')
        return matching_nodes[0]

    def to_nx(self):
        """
        Transforms the current graph into a networkx graph
        :return: networkx graph
        """
        graph_nx = nx.Graph()
        for node in self.get_nodes():
            part = node.get_part()
            info = f'\nPartID={part.get_part_id()}\nFamilyID={part.get_family_id()}'

            graph_nx.add_node(node, info=info)

        for source_node in self.get_nodes():
            connected_nodes = self.get_edges()[source_node]
            for connected_node in connected_nodes:
                graph_nx.add_edge(source_node, connected_node)
        assert graph_nx.number_of_nodes() == len(self.get_nodes())
        return graph_nx

    def draw(self):
        """ Draws the graph with NetworkX and displays it. """
        plt.clf()
        graph_nx = self.to_nx()
        labels = nx.get_node_attributes(graph_nx, 'info')
        nx.draw(graph_nx, labels=labels)
        plt.show()
        if TEMPORARY_VISUALIZATION_PATH is not None:
            plt.savefig(TEMPORARY_VISUALIZATION_PATH)

    def get_edges(self) -> Dict[Node, List[Node]]:
        """
        Returns a dictionary containing all directed edges.
        :return: dict of directed edges
        """
        return self.__edges

    def get_nodes(self) -> Set[Node]:
        """
        Returns a set of all nodes.
        :return: set of all nodes
        """
        return self.__nodes

    def get_parts(self) -> Set[Part]:
        """
        Returns a set of all parts of the graph.
        :return: set of all parts
        """
        return {node.get_part() for node in self.get_nodes()}

    def get_construction_id(self) -> int:
        """
        Returns the unix timestamp for the creation date of the corresponding construction.
        :return: construction id (aka creation timestamp)
        """
        return self.__construction_id

    def __breadth_search(self, start_node: Node) -> List[Node]:
        """
        Performs a breadth search starting from the given node and returns all node it has seen
        (including duplicates due to cycles within the graph).
        :param start_node: Node of the graph to start the search
        :return: list of all seen nodes (may include duplicates)
        """
        parent_node: Node = None
        queue: List[Tuple[Node, Node]] = [(start_node, parent_node)]
        seen_nodes: List[Node] = [start_node]
        while queue:
            curr_node, parent_node = queue.pop()
            new_neighbors: List[Node] = [n for n in self.get_edges().get(curr_node) if n != parent_node]
            queue.extend([(n, curr_node) for n in new_neighbors if n not in seen_nodes])
            seen_nodes.extend(new_neighbors)
        return seen_nodes

    def is_connected(self) -> bool:
        """
        Returns a boolean that indicates if the graph is connected
        :return: boolean if the graph is connected
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__is_connected is None:
            # choose random start node and start breadth search
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            # if we saw all nodes during the breadth search, the graph is connected
            self.__is_connected = set(seen_nodes) == self.get_nodes()
        return self.__is_connected

    def is_cyclic(self) -> bool:
        """
        Returns a boolean that indicates if the graph contains at least one non-trivial cycle.
        A bidirectional edge between two nodes is a trivial cycle, so only cycles of at least three nodes make
        a graph cyclic.
        :return: boolean if the graph contains a cycle
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__contains_cycle is None:
            # choose random start node and start breadth search
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            # graph contains a cycle if we saw a node twice during breadth search
            self.__contains_cycle = len(seen_nodes) != len(set(seen_nodes))
        return self.__contains_cycle

    def get_adjacency_matrix(self, part_order: Tuple[Part]) -> np.ndarray:
        """
        Returns a full adjacency matrix in the given order of parts
        :param part_order:
        :return:
        """
        size = len(part_order)
        adj_matrix = np.zeros((size, size), dtype=int)
        edges: Dict[Node, List[Node]] = self.get_edges()

        for idx, part in enumerate(part_order):
            node = self.__get_node_for_part(part)

            for idx2, part2 in enumerate(part_order):
                node2 = self.__get_node_for_part(part2)

                if node2 in edges[node]:
                    adj_matrix[idx, idx2] = adj_matrix[idx2, idx] = 1

        return adj_matrix

    @classmethod
    def from_adjacency_matrix(cls, part_list: List[Part], adjacency_matrix: np.ndarray):
        """
        Creates the graph from a given list of parts and an adjacency matrix in the order of the 
        parts list.
        """
        if len(adjacency_matrix.shape) == 3:
            adjacency_matrix = adjacency_matrix[0]

        graph: Graph = Graph(datetime.now())
        # node_list = [Node(i, part) for (i, part) in enumerate(part_list)]
        for part in part_list:
            graph.add_node_without_edge(part)
        
        for row in range(len(part_list)):
            for col in range(len(part_list)):
                if adjacency_matrix[row][col] == 1:
                    graph.add_undirected_edge(part_list[row], part_list[col])
        
        return graph

    def get_nonredundand_connections_array(self, part_order: Tuple[Part], pad_to_node_count = None,  padding_value: int = -1) -> List:
        """
        Returns an array representing non-redundant parts of the adjacency matrix in the given order of parts
        Optionally, the list is padded to a fixed count of nodes with the defined values.

        Example:
        part_order=[A, B, C], pad_to_node_count=4, padding_value=-1:
        Adjacency matrix indices:
           A | B | C | ?
        A  0   1   2   3 
        B  4   5   6   7
        C  8   9   10  11
        ?  12  13  14  15
        Content indices of the reduced array:
        [1, 2, 3, 6, 7, 11]
       
        """
        assert pad_to_node_count is None or pad_to_node_count >= len(part_order), \
            "Can not pad to a node amount smaller than the acutal count"

        reduzed_connections_array =  []
        edges: Dict[Node, List[Node]] = self.get_edges()
        padded_parts_len = len(part_order) if pad_to_node_count is None else pad_to_node_count

        # Iterate through rows:
        for i in range(padded_parts_len - 1):
            if i >= len(part_order):
                # In padding area
                for j in range(i + 1, padded_parts_len):
                    reduzed_connections_array.append(padding_value)
                continue

            i_node = self.__get_node_for_part(part_order[i])
             
            # Iterate through columns
            for j in range(i + 1, padded_parts_len):
                if j >= len(part_order):
                    # In padding area
                    reduzed_connections_array.append(padding_value)
                    continue

                j_node = self.__get_node_for_part(part_order[j])
                if j_node in edges[i_node]:
                    reduzed_connections_array.append(1)
                else:
                    reduzed_connections_array.append(0)

        return reduzed_connections_array

    @classmethod
    def from_nonredundand_connections_array(cls, part_order: Tuple[Part], reduzed_connections_array: List, pad_to_node_count = None):
        """
        Creates the graph from a given list of parts and an array representation of the 
        non-redundant parts of the adjacency matrix in the given order of parts.
        A padding can be considered.
        """
        assert pad_to_node_count is None or pad_to_node_count >= len(part_order), \
            "Can not use a padding smaller than the acutal count"

        graph: Graph = Graph(datetime.now())
        padded_parts_len = len(part_order) if pad_to_node_count is None else pad_to_node_count

        for part in part_order:
            graph.add_node_without_edge(part)

        # Current reduced index:
        i = 0

        # Iterate through rows:
        for row in range(padded_parts_len - 1):
            if row >= len(part_order):
                # In padding area
                i += padded_parts_len - (row + 1)
                continue
             
            # Iterate through columns
            for col in range(row + 1, padded_parts_len):
                if col >= len(part_order):
                    # In padding area
                    i += 1
                    continue

                if reduzed_connections_array[i] == 1:
                    graph.add_undirected_edge(part_order[row], part_order[col])

                i += 1
        
        
        return graph