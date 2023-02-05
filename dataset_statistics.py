"""
This script computes some statistics about the given data, 
like the maximum part_id and family_id
"""

from dataset_retriever import DatasetRetriever
from graph import Graph
from node import Node
from part import Part


if __name__ == '__main__':
    dataset_retriever = DatasetRetriever.instance()
    graphs = dataset_retriever.all_graphs

    first_part: Part = graphs[0].get_nodes().pop().get_part()
    min_part_id = first_part.get_part_id()
    max_part_id = first_part.get_part_id()
    min_family_id = first_part.get_family_id()
    max_family_id = first_part.get_family_id()
    max_amount_of_nodes = len(graphs[0].get_nodes())

    graph: Graph
    for graph in graphs:
        nodes = graph.get_nodes()
        max_amount_of_nodes = max(len(nodes), max_amount_of_nodes)
        node: Node
        for node in nodes:
            part = node.get_part()

            min_part_id = min(part.get_part_id(), min_part_id)
            max_part_id = max(part.get_part_id(), max_part_id)
            min_family_id = min(part.get_family_id(), min_family_id)
            max_family_id = max(part.get_family_id(), max_family_id)

    print(f"Min part id: {min_part_id}")
    print(f"Max part id: {max_part_id}")
    print(f"Min family id: {min_family_id}")
    print(f"Max family id: {max_family_id}")
    print(f"Max amount of nodes: {max_amount_of_nodes}")