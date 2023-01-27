from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
from enum import Enum
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.functional import pad

from graph import Graph
from node import Node
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
import mlflow

MAX_NUMBER_OF_PARTS_PER_GRAPH = 30

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    # TODO: maybe add as second base class to pred model
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 30*30),
            nn.InstanceNorm1d(30*30)
        )
        self.unflatten = nn.Unflatten(1, (30, 30))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        y = self.unflatten(logits)
        return y

class CustomGraphDataset(Dataset):
    def __init__(self, graphs: List[Graph]):
        self.graphs = graphs
        self.parts_lists = [list(graph.get_parts()) for graph in self.graphs]
        # Sort the parts to reduce possible combinations
        for parts_list in self.parts_lists:
            parts_list.sort()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
    
        parts_tensor = torch.tensor([[part.get_part_id(), part.get_family_id()] for part in self.parts_lists[idx]], dtype=torch.float32)
        adj_matr_tensor = torch.tensor(self.graphs[idx].get_adjacency_matrix(part_order=self.parts_lists[idx]), dtype=torch.float32)

        # Padding to achieve the same size for each input
        missing_node_count = MAX_NUMBER_OF_PARTS_PER_GRAPH - len(self.parts_lists[idx])
        if missing_node_count > 0:
            parts_tensor = pad(parts_tensor, (0, 0, 0, missing_node_count), "constant", -1)
            adj_matr_tensor = pad(adj_matr_tensor, (0, missing_node_count, 0, missing_node_count), "constant", -1)

        return parts_tensor, adj_matr_tensor



class NeuralNetworkPredictionModel(BasePredictionModel):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """
    def __init__(self):
        self._model: NeuralNetwork = NeuralNetwork().to(device)
        self._loss_fn = nn.SmoothL1Loss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3)
        print("Using model:")
        print(self._model)

    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        parts_list = list(parts)
        # Sort the parts to reduce possible combinations
        parts_list.sort()

        parts_tensor = torch.tensor([[[part.get_part_id(), part.get_family_id()] for part in parts_list]], dtype=torch.float32)

        # Padding to achieve the same size for each input
        missing_node_count = MAX_NUMBER_OF_PARTS_PER_GRAPH - len(parts_list)
        if missing_node_count > 0:
            parts_tensor = pad(parts_tensor, (0, 0, 0, missing_node_count), "constant", -1)


        self._model.eval()
        with torch.no_grad():
            X = parts_tensor.to(device)
            pred = self._model(X)

        # TODO: create actual graph from adjacency matrix instead of this pseudo output
        pred_thresh = torch.where(pred > 0, 1, 0)
        graph = Graph.from_adjacency_matrix(part_list=parts_list, adjacency_matrix=pred_thresh)

        # Pseudo output:
        # graph = Graph()
        # parts_list = list(parts) 
        # for i in range(len(parts_list) - 1):
        #     graph.add_undirected_edge(parts_list[i], parts_list[i+1])
    
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
    def train_new_instance(cls, train_graphs: List[Graph]):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        new_instance = cls()
        print("Loading training data...")
        graphs_dataset = CustomGraphDataset(train_graphs)
        graphs_dataloader = DataLoader(graphs_dataset, batch_size=64, shuffle=True)
        # graphs_dataloader = test_dataloader
        
        print("Starting training...")
        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            new_instance._train(dataloader=graphs_dataloader)
                
        return new_instance


    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        torch.save(self._model.state_dict(), file_path)

    def _train(self, dataloader):
        size = len(dataloader.dataset)
        self._model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self._model(X)
            loss = self._loss_fn(pred, y)
            mlflow.log_metric("loss", loss)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")