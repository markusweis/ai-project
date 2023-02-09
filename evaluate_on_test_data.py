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
from evaluation import evaluate, evaluate_edge_accuracy
from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.base_neural_network.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class


# SELECTED_MODEL_TYPE = PredictionModels.STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL.value

# SELECTED_MODEL_TYPE = PredictionModels.NEURAL_NETWORK_PREDICTION_MODEL.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/BASE_DNN.pth"

SELECTED_MODEL_TYPE = PredictionModels.GNN.value
SELECTED_MODEL_PATH = "prediction_models/model_instances/GNN2.pth"

def load_model(file_path: str, model_type: str = SELECTED_MODEL_TYPE):
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    model_class = get_model_class(model_type)
    return model_class.load_from_file(file_path=file_path)


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation (Select via parameter or constant)



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
    prediction_model = load_model(
        model_file_path, model_type=model_type)

    #prediction_model.debug_predict_graph(dataset_retriever.get_evaluation_graphs()[:20])

    print("Evaluating the model...")
    eval_score = evaluate_edge_accuracy(prediction_model, dataset_retriever.get_test_graphs())
    print(f"Evaluation edge accuracy score on the test dataset: {eval_score}")
