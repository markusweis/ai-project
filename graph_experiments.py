
from graph import Graph
from node import Node
from part import Part

graph = Graph()

part1 = Part(part_id=3, family_id=3)
part2 = Part(1, 1)
part3 = Part(1, 1)
graph.add_undirected_edge(part1, part2)

# node = Node(0, part3)
# graph.add_node(node)
# graph.__edges[node] = []
graph.add_node_without_edge(part3)

graph.draw()