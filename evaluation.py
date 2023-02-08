"""
Provides edge accuracy evaluations. 

Can be used on subsets of the data with evaluate_edge_accuracy(model, graphs: np.array)

If this file is run as main, the score on the test-dataset is evaluated.
"""


from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import sys
from dataset_retriever import DatasetRetriever
from tqdm import tqdm
import lovely_tensors as lt

import mlflow
from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.base_neural_network.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class


# SELECTED_MODEL_TYPE = PredictionModels.STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL.value


SELECTED_MODEL_TYPE = PredictionModels.GNN.value
SELECTED_MODEL_PATH = "prediction_models/model_instances/GNN.pth"

def load_model(file_path: str, model_type: str = SELECTED_MODEL_TYPE):
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    model_class = get_model_class(model_type)
    return model_class.load_from_file(file_path=file_path)


def evaluate(model: BasePredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> float:
    """
    Calculates the edge accuracy on the given graphs subset.

    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0

    progress_bar = tqdm(data_set) # Wraps progress bar around an interable 
    for input_parts, target_graph in progress_bar:
        predicted_graph = model.predict_graph(input_parts)

        #target_graph.draw()
        #predicted_graph.draw()

        edges_counter += len(input_parts) * len(input_parts)
        sum_correct_edges += edge_accuracy(predicted_graph, target_graph)

        # FYI: maybe some more evaluation metrics will be used in final evaluation

    return sum_correct_edges / edges_counter * 100


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(
        target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(
    ), 'Mismatch in expected and given parts.'

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(
        predicted_graph.get_parts())

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [
        pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [
        next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [
            t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all([len(perm) == len(parts) for perm in full_perms]
               ), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation (Select via parameter or constant)

def evaluate_edge_accuracy(model, graphs: np.array) -> float:
    instances = [(graph.get_parts(), graph) for graph in graphs]
    return evaluate(model, instances)

if __name__ == '__main__':
    """
    Loads the model at SELECTED_MODEL_PATH / argument 2 of type SELECTED_MODEL_TYPE / argument 1.
    Calculates the edge accuracy for the test dataset.
    """
    lt.monkey_patch()

    # Load data
    dataset_retriever = DatasetRetriever.instance()

    # Load the model
    print("Loading the model...")
    model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    model_file_path = SELECTED_MODEL_PATH if len(sys.argv) < 3 else sys.argv[2]
    prediction_model: BasePredictionModel = load_model(
        model_file_path, model_type=model_type)

    print("Evaluating the model...")
    eval_score = evaluate_edge_accuracy(prediction_model, dataset_retriever.get_test_graphs())
    print(f"Evaluation edge accuracy score on the test dataset: {eval_score}")
