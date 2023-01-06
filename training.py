from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import sys

from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class


SELECTED_MODEL_TYPE = PredictionModels.NEURAL_NETWORK_PREDICTION_MODEL.value
SELECTED_MODEL_PATH = "prediction_models/model_instances/test_model.pth"


# ---------------------------------------------------------------------------------------------------------------------
# Train a selected model (Select via parameter or constant):

if __name__ == '__main__':
    # Load train data
    with open('data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    # Load the model class and train
    model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    model_file_path = SELECTED_MODEL_PATH if len(sys.argv) < 3 else sys.argv[2]

    model_class = get_model_class(model_type)
    new_model_instance: BasePredictionModel = model_class.train_new_instance(train_graphs=train_graphs[100:])
    new_model_instance.store_model(model_file_path)