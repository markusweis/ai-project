from itertools import permutations
import os
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import sys
import mlflow
import lovely_tensors as lt
from dataset_retriever import DatasetRetriever

from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.base_neural_network.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class
from evaluation import evaluate_edge_accuracy

# SELECTED_MODEL_PATH = None  # Not needed anymore thanks to mlflow. Can still be used, though.

# SELECTED_MODEL_TYPE = PredictionModels.GGN.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/GNN.pth"

SELECTED_MODEL_TYPE = PredictionModels.GNN.value
SELECTED_MODEL_PATH = "prediction_models/model_instances/GNN.pth"

# SELECTED_MODEL_TYPE = PredictionModels.STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL.value
# SELECTED_MODEL_PATH = "prediction_models/model_instances/PSEUDO_MODEL.pth"

# ---------------------------------------------------------------------------------------------------------------------
# Train a selected model (Select via parameter or constant):

if __name__ == '__main__':
    """
    Trains a model of the type SELECTED_MODEL_TYPE / argument 1 and logs it to mlflow.
    If either SELECTED_MODEL_PATH or argument 2 is not empty, the model will be stored there, too.
    """
    # make tensor output more readable 
    lt.monkey_patch()
    # Load data
    dataset_retriever = DatasetRetriever.instance()

    # Load the model class
    model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    model_file_path = SELECTED_MODEL_PATH if len(sys.argv) < 3 else sys.argv[2]

    model_class = get_model_class(model_type)

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=model_class.get_name()):
        # Train the new model:
        new_model_instance = model_class.train_new_instance(
                train_set=dataset_retriever.get_training_graphs()[:100],  #TODO: remove 
                val_set=dataset_retriever.get_evaluation_graphs())
        
        # Evaluate the final edge accuracies on both the original training data and the evaluation data
        # print("Calculating edge accuracy on training data:")
        # edge_acc_training = evaluate_edge_accuracy(new_model_instance, dataset_retriever.get_training_graphs())
        # print(f"Evaluation edge accuracy score on the training dataset: {edge_acc_training}")
        # -> Removed due to some graphs being to large for the given edge accuracy calculations!
        if model_file_path is not None and model_file_path != "":
            new_model_instance.store_model(model_file_path)


        print("Calculating edge accuracy on evaluation data:")
        edge_acc_evaluation = evaluate_edge_accuracy(new_model_instance, dataset_retriever.get_evaluation_graphs())
        print(f"Evaluation edge accuracy score on the evaluation dataset: {edge_acc_evaluation}")

        # Store the model without mlflow, if path is given
        
        # Log parameters and metrics
        mlflow.log_params(new_model_instance.get_meta_params())
        # mlflow.log_metric("edge_acc_training", edge_acc_training) 
        # Removed due to some graphs being to large for the given edge accuracy calculations!
        mlflow.log_metric("edge_acc_evaluation", edge_acc_evaluation)
        
        # Log the model to mlflow
        new_model_instance.log_pytorch_models_to_mlflow()

        # Alternative without using the mlflow pytorch module:
        # mlflow.log_artifact(
        #     model_file_path,
        #     ""
        # )
        
        pass

