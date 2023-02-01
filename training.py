from itertools import permutations
import os
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import sys
import mlflow

from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.base_neural_network.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class
from evaluation import eval_model_on_train_set

SELECTED_MODEL_PATH = None  # Not needed anymore thanks to mlflow. Can still be used, though.

# SELECTED_MODEL_TYPE = PredictionModels.GGN.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/GNN.pth"

SELECTED_MODEL_TYPE = PredictionModels.NEURAL_NETWORK_PREDICTION_MODEL.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/BASE_DNN.pth"

# SELECTED_MODEL_TYPE = PredictionModels.STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/PSEUDO_MODEL.pth"

# ---------------------------------------------------------------------------------------------------------------------
# Train a selected model (Select via parameter or constant):

if __name__ == '__main__':
    # Load data
    print("Loading data...")
    with open('data/graphs.dat', 'rb') as file:
        graphs = np.asarray(pickle.load(file))
    print("Done.")

    # train, validation and test split
    np.random.seed(42)
    idxs = np.arange(len(graphs))
    idxs = np.random.permutation(idxs)
    val_size = int(len(idxs) * 0.05)
    test_size = int(len(idxs) * 0.05)
    train_size = len(idxs) - val_size - test_size

    train_set = graphs[idxs[:train_size]]
    val_set = graphs[idxs[train_size:(train_size+val_size)]]
    test_set = graphs[idxs[(train_size + val_size):]]

    # Load the model class
    model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    model_file_path = SELECTED_MODEL_PATH if len(sys.argv) < 3 else sys.argv[2]

    model_class = get_model_class(model_type)

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=model_class.get_name()):
        new_model_instance = model_class.train_new_instance(
                train_set=train_set, val_set=val_set)
        
        score = eval_model_on_train_set(new_model_instance) # TODO: Why eval on train? 

        # Store the model without mlflow, if path is given
        if model_file_path is not None and model_file_path != "":
            new_model_instance.store_model(model_file_path)
        
        # Log parameters and metrics
        mlflow.log_params(new_model_instance.get_meta_params())
        mlflow.log_metric("edge_acc", score)

        # Log the model to mlflow
        mlflow.pytorch.log_model(
            new_model_instance.model,
            model_class.get_name()
        )

        # Alternative without using the mlflow pytorch module:
        # mlflow.log_artifact(
        #     model_file_path,
        #     ""
        # )
        
