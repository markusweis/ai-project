from itertools import permutations
import os
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import sys
import mlflow
import lovely_tensors as lt
from dataset_retriever import DatasetRetriever
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from graph import Graph
from part import Part
from functools import partial
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.base_neural_network.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.prediction_models_enum import PredictionModels, get_model_class
from evaluation import evaluate_edge_accuracy


SELECTED_MODEL_TYPE = PredictionModels.GNN.value

DATA_PATH = 'data/graphs.dat'

NUM_SAMPLES = 20

config = {
        "NUM_GNN_LAYERS": tune.lograndint(1, 20),
        "EMBDEDDING_FEATURES": tune.sample_from(lambda _: 2 ** np.random.randint(4, 11)),

        "NUM_FC_LAYERS": tune.lograndint(1, 20),
        "FC_FEATURES": tune.sample_from(lambda _: 2 ** np.random.randint(4, 11)),
        
        "LEARNING_RATE": tune.loguniform(1e-4, 1e-1),
        "WD": tune.choice([0, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        "DROPOUT": tune.choice([0, 0.1, 0.2, 0.4])
    }

if __name__ == '__main__':
    
    # make tensor output more readab+le 
    lt.monkey_patch()
    # Load data
    # dataset_retriever = DatasetRetriever.instance()

    # Load the model class
    model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    model_file_path = DATA_PATH if len(sys.argv) < 3 else sys.argv[2]

    model_class = get_model_class(model_type)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["NUM_GNN_LAYERS", "EMBDEDDING_FEATURES", "NUM_FC_LAYERS",
         "FC_FEATURES", "LEARNING_RATE", "WEIGHT_DECAY", "DROPOUT"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(model_class.train_new_instance, data_dir=DATA_PATH),
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    pass
        
    # Train the new model:
    # new_model_instance = model_class.train_new_instance(
    #         train_set=dataset_retriever.get_training_graphs(), 
    #         val_set=dataset_retriever.get_validation_graphs())
    
    # else: 
    #     print("Loading the model and continueing training ...")
    #     model_type = SELECTED_MODEL_TYPE if len(sys.argv) < 2 else sys.argv[1]
    #     model_file_path = DATA_PATH if len(sys.argv) < 3 else sys.argv[2]
    #     new_model_instance = load_model(file_path=DATA_PATH, model_type=SELECTED_MODEL_TYPE)
    #     new_model_instance.continue_training(dataset_retriever.get_training_graphs(), dataset_retriever.get_validation_graphs())
    # # Evaluate the final edge accuracies on both the original training data and the evaluation data
    # # print("Calculating edge accuracy on training data:")
    # # edge_acc_training = evaluate_edge_accuracy(new_model_instance, dataset_retriever.get_training_graphs())
    # # print(f"Evaluation edge accuracy score on the training dataset: {edge_acc_training}")
    # # -> Removed due to some graphs being to large for the given edge accuracy calculations!


    # print("Calculating edge accuracy on evaluation data:")
    # edge_acc_evaluation = evaluate_edge_accuracy(new_model_instance, dataset_retriever.get_validation_graphs())
    # print(f"Evaluation edge accuracy score on the evaluation dataset: {edge_acc_evaluation}")

    
    
    # # mlflow.log_metric("edge_acc_training", edge_acc_training) 
    # # Removed due to some graphs being to large for the given edge accuracy calculations!
    # mlflow.log_metric("edge_acc_evaluation", edge_acc_evaluation)
    
    # # Log the model to mlflow
    # new_model_instance.log_pytorch_models_to_mlflow()

    # # Alternative without using the mlflow pytorch module:
    # # mlflow.log_artifact(
    # #     model_file_path,
    # #     ""
    # # )
    
    # pass



