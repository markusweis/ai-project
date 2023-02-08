import pickle
from enum import Enum
from itertools import permutations
from typing import Dict, List, Set, Tuple

import mlflow
import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from graph import Graph
from node import Node
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.gnn.meta_parameters import *
from prediction_models.gnn.dataset import CustomGraphDataset
from torch_geometric.utils import negative_sampling
from prediction_models.gnn import meta_parameters
from prediction_models.gnn.gnn_module import GNNModel



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class GNNPredictionModel(BasePredictionModel):
    """wraps the GNN Model and LinkPredictor for saving, training, ..."""

    def __init__(self, path):
        self.path = path
        self._model: GNNModel = GNNModel(
            (2 * (MAX_SUPPORTED_PART_ID + 1)),  # F
            EMBDEDDING_FEATURES,
            NUM_GNN_LAYERS,
            FC_FEATURES,
            NUM_FC_LAYERS,
            DROPOUT
        ).to(device)
        
        self._optimizer = torch.optim.Adam(
            list(self._model.parameters()),
            lr=LEARNING_RATE, weight_decay=WD
        )

        self._loss = nn.BCELoss()
    
    @classmethod
    def get_name(self) -> str:
        """
        :return: Name of the model (used as model and run name)
        """
        return "gnn_model"

    @classmethod
    def get_meta_params(self) -> dict:
        """
        :return: Dict containing all used meta parameters
        """
        return {
            "MAX_NUMBER_OF_PARTS_PER_GRAPH": meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH,
            "EMBDEDDING_DIMS": meta_parameters.EMBDEDDING_DIMS,
            "HIDDEN_LAYERS_SIZE": meta_parameters.HIDDEN_LAYERS_SIZE,
            "LEARNING_RATE": meta_parameters.LEARNING_RATE,
            "DROPOUT": meta_parameters.DROPOUT
        }

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        loaded_instance = cls()
        loaded_instance._model.state_dict(torch.load(file_path))
        return loaded_instance

    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        torch.save(self._model.state_dict(), file_path)

    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """

        #setup input 
        parts_list = list(parts)
        parts_list.sort()
        parts_tensor = torch.tensor(
             [(part.get_part_id(), part.get_family_id()) for part in parts_list])
        parts_tensor = torch.nn.functional.one_hot(parts_tensor, 
            MAX_SUPPORTED_PART_ID + 1).float()

        self._model.eval()
        with torch.no_grad():
            prediction = self._model(torch.flatten(parts_tensor, start_dim=1))


        edge_list = edge_selection_strategy(prediction, len(parts_list))
        # Pseudo output:
        graph = Graph()
        parts_list = list(parts) 
        for part in parts_list:
            graph.add_node_without_edge(part)
        for edge in edge_list:
            graph.add_undirected_edge(parts_list[edge[0]], parts_list[edge[1]])
    
        return graph


    


    @classmethod
    def train_new_instance(cls, path,  train_set: np.ndarray, val_set: np.ndarray):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        new_instance = cls(path)
        train_dataset = CustomGraphDataset(train_set)
        val_dataset = CustomGraphDataset(val_set)


        print("Starting training...")
        epochs = 8
        mlflow.log_param("epochs", epochs)
        for t in range(epochs):
            print(f"Epoch {t+1}")
            epoch_loss = new_instance._train(train_dataset)

            print(f"Epoch {t+1}: Mean loss: {epoch_loss}")


            # loss on validation set
            loss = 0
            new_instance._model.eval()
            for features, labels in val_dataset:
                features, labels = features.to(device), labels.to(device)
                curr_loss = new_instance._loss(new_instance._model(features), labels)
                loss += curr_loss
                new_instance.store_model(new_instance.path)
            # is the normlaization correct?
            normalized_val_loss = loss / (len(val_set))
            mlflow.log_metric("val_loss", normalized_val_loss,
                              (t + 1) * len(train_set))
            print(f"Validation loss: {normalized_val_loss}")
        return new_instance

    
        

    def _train(self, dataset: CustomGraphDataset):
        """
        Trains for a single epoch. 
        Runs offline training for model, link_predictor and node embeddings
        edges and supervision edges.
        :param dataloader: The custom dataloader for the test set. 
        :param emb_model: Torch Graph model used for updating node embeddings based on message passing
        :param link_predictor: Torch model used for predicting whether edge exists or not
        :param emb: (N, d) Initial node embeddings for all N nodes in graph
        :param optimizer: Torch Optimizer to update model parameters
        :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
        """

        self._model.train()

        train_losses = []
        
        progress_bar = tqdm(dataset) # Wraps progress bar around an interable 

        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            self._optimizer.zero_grad()

            preds = self._model(features)
            loss = self._loss(preds, labels)

            loss.backward()
            self._optimizer.step()

            train_losses.append(loss.item())
            mlflow.log_metric("train_loss",str(loss.item()))
            # print(loss.item())

            progress_bar.set_description(str(loss.item()))
            progress_bar.update()

        return sum(train_losses) / len(train_losses)

    def log_pytorch_models_to_mlflow(self):
        """
        Logs the model or models to mlflow
        """
        # Log the model to mlflow
        mlflow.pytorch.log_model(
            self.model,
            self.get_name()
        )

def edge_selection_strategy(preds, num_nodes):
    """
        for a flatt array of predictions, returns a list of edges (E, 2)
        Select n highest scoreing edges 
    """
    selected_edges_in_flat_array = torch.topk(preds, num_nodes).indices
    all_edge_indices = torch.triu_indices(num_nodes, num_nodes, 1)
    transposed = all_edge_indices.t()

    selected_edges = transposed[selected_edges_in_flat_array]
    return selected_edges.t()
