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
from prediction_models.gnn.link_predictor import LinkPredictor
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.gnn.constants import *
from prediction_models.gnn.dataset import CustomGraphDataset
from torch_geometric.utils import negative_sampling

from prediction_models.gnn.gnn_stack import GNNStack



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class GNNPredictionModel(BasePredictionModel):
    """wraps the GNN Model and LinkPredictor for saving, training, ..."""

    def __init__(self):
        self._embeddings_model: GNNStack = GNNStack(
            2,  # F
            HIDDEN_DIMS,
            HIDDEN_DIMS, 
            NUM_LAYERS,
            DROPOUT
        ).to(device)
        self._link_predictor: LinkPredictor = LinkPredictor(
            HIDDEN_DIMS,  # TODO: seperate constants; don't have to be same
            HIDDEN_DIMS,  
            NUM_LAYERS,
            DROPOUT
        ).to(device)
        self._embeddings: nn.Embedding = nn.Embedding(
            MAX_NUMBER_OF_PARTS_PER_GRAPH,
            EMBDEDDING_DIMS ).to(device) 

        self._optimizer = torch.optim.Adam(
            list(self._embeddings_model.parameters()) + 
            list(self._link_predictor.parameters()) + 
            list(self._embeddings.parameters()),
            lr=LR, weight_decay=WD
        )

    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        parts_list = list(parts)
        # Sort the parts to reduce possible combinations
        parts_list.sort()

        parts_tensor = torch.tensor(
            [[[part.get_part_id(), part.get_family_id()] for part in parts_list]], dtype=torch.float32)

        # Padding to achieve the same size for each input
        missing_node_count = MAX_NUMBER_OF_PARTS_PER_GRAPH - len(parts_list)
        if missing_node_count > 0:
            parts_tensor = pad(
                parts_tensor, (0, 0, 0, missing_node_count), "constant", -1)

        self._model.eval()
        with torch.no_grad():
            X = parts_tensor.to(device)
            pred = self._model(X)

        pred_thresh = torch.where(pred > 0, 1, 0)
        graph = Graph.from_adjacency_matrix(
            part_list=parts_list, adjacency_matrix=pred_thresh)

        return graph

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

    @classmethod
    def train_new_instance(cls, train_set: np.ndarray, val_set: np.ndarray):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        new_instance = cls()
        train_dataset = CustomGraphDataset(train_set)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True)

        val_dataset = CustomGraphDataset(val_set)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        print("Starting training...")
        epochs = 8
        mlflow.log_param("epochs", epochs)
        for t in range(epochs):
            print(f"Epoch {t+1}")
            new_instance._train(train_dataloader,
                 new_instance._embeddings_model, new_instance._link_predictor,
                 new_instance._embeddings.weight, new_instance._optimizer)

            # loss on validation set
            loss = 0
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = new_instance._model(X)
                loss += new_instance._loss_fn(pred, y)
            # is the normlaization correct?
            normalized_val_loss = loss / (len(val_set) / 64)
            mlflow.log_metric("val loss", normalized_val_loss,
                              (t + 1) * len(train_set))
            print(f"validation loss: {normalized_val_loss}")
        return new_instance

    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        torch.save(self._model.state_dict(), file_path)

    def _train(self, dataloader: CustomGraphDataset, emb_model: GNNStack, link_predictor, emb, optimizer):
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

        self._embeddings_model.train()
        self._link_predictor.train()

        train_losses = []

        for parts_list, edge_index in dataloader:
            optimizer.zero_grad()
            
             # batch size of 1 makes 3 dims but emb_model expects 2
            parts_list = parts_list[0]
            edge_index = edge_index[0]

            # Run message passing on the inital node embeddings to get updated embeddings
            node_emb = emb_model(parts_list, edge_index)  # (N, d)

            # Predict the class probabilities on the batch of positive edges using link_predictor
            pos_edge = edge_index  # (2, E)
            pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (E, )

            # Sample negative edges (same as number of positive edges (if possible)) and predict class probabilities
            # num_neg_samples= None -> same number as E 
            neg_edge = negative_sampling(edge_index, num_nodes=emb.shape[0],
                                        num_neg_samples=None, method='dense')  # (2, E)

                                                                                
            neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (E, )

            # Compute the corresponding negative log likelihood loss on the positive and negative edges
            loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            mlflow.log_metric("train_loss",str(loss.item()))
            # print(loss.item())

        return sum(train_losses) / len(train_losses)
