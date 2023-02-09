import os
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
from dataset_retriever import DatasetRetriever
from ray import tune
from evaluation import evaluate_edge_accuracy
from graph import Graph
from node import Node
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel
from prediction_models.gnn.dataset import CustomGraphDataset
from torch_geometric.utils import negative_sampling
from prediction_models.gnn import meta_parameters
from prediction_models.gnn.gnn_module import GNNModel

INTERMEDIATE_MODEL_STORE_PATH = "prediction_models/model_instances/GNN1.pth"

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class GNNPredictionModel(BasePredictionModel):
    """wraps the GNN Model and LinkPredictor for saving, training, ..."""

    def __init__(self, config_override: dict = None):
        # Meta parameter configuration:
        self.num_gnn_layers = meta_parameters.NUM_GNN_LAYERS
        self.embedding_features = meta_parameters.EMBDEDDING_FEATURES
        self.num_fc_layers = meta_parameters.NUM_FC_LAYERS
        self.fc_features = meta_parameters.FC_FEATURES
        self.learning_rate = meta_parameters.LEARNING_RATE
        self.weight_decay = meta_parameters.WEIGHT_DECAY
        self.dropout = meta_parameters.DROPOUT
        if config_override is not None:
            self.num_gnn_layers = config_override.get("NUM_GNN_LAYERS")
            self.embedding_features = config_override.get("EMBDEDDING_FEATURES")
            self.num_fc_layers = config_override.get("NUM_FC_LAYERS")
            self.fc_features = config_override.get("FC_FEATURES")
            self.learning_rate = config_override.get("LEARNING_RATE")
            self.weight_decay = config_override.get("WEIGHT_DECAY")
            self.dropout = config_override.get("DROPOUT")

        self.learning_epochs = meta_parameters.LEARNING_EPOCHS

        # Instance
        self.step = 0 
        self.epoch = 1
        self._model: GNNModel = GNNModel(
            (2 * (meta_parameters.MAX_SUPPORTED_PART_ID + 1)),  # F
            self.embedding_features,
            self.num_gnn_layers,
            self.fc_features,
            self.num_fc_layers,
            self.dropout
        ).to(device)
        
        self._optimizer = torch.optim.Adam(
            list(self._model.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay
        )

        self._loss = nn.BCELoss()
    
    @classmethod
    def get_name(self) -> str:
        """
        :return: Name of the model (used as model and run name)
        """
        return "gnn_model"

    def get_meta_params(self) -> dict:
        """
        :return: Dict containing all used meta parameters
        """
        return {
            "NUM_GNN_LAYERS": self.num_gnn_layers,
            "EMBDEDDING_FEATURES": self.embedding_features,
            "NUM_FC_LAYERS": self.num_fc_layers,
            "HIDDEN_LAYERS_SIZE": self.fc_features,
            "LEARNING_RATE": self.learning_rate,
            "WEIGHT_DECAY": self.weight_decay,
            "DROPOUT": self.dropout,
            "LEARNING_EPOCHS": self.learning_epochs
        }

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        loaded_instance = cls()
        loaded_instance._model.to(device)
        checkpoint = torch.load(file_path)
        loaded_instance._model.load_state_dict(checkpoint)
        return loaded_instance

    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        torch.save(self._model.state_dict(), file_path)


    def debug_predict_graph(self, graphs: List[Graph]):
        ds = CustomGraphDataset(graphs)
        for ((features, labels), graph) in zip(ds, graphs):
            predicted_graph = self.predict_graph(graph.get_parts())
            preds = self._model(features.to(device))

            print("eh")

    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """

        #setup input 
        parts_list = list(parts)
        parts_tensor = torch.tensor(
             [(part.get_part_id(), part.get_family_id()) for part in parts_list])
        parts_tensor = torch.nn.functional.one_hot(parts_tensor, 
            meta_parameters.MAX_SUPPORTED_PART_ID + 1).float().to(device)
        parts_tensor = torch.flatten(parts_tensor, start_dim=1)
        self._model.eval()
        with torch.no_grad():
            prediction = self._model(parts_tensor)


        edge_list = edge_selection_strategy(prediction, len(parts_list))

        graph = Graph()
        for part in parts_list:
            graph.add_node_without_edge(part)
        for edge in edge_list:
            graph.add_undirected_edge(parts_list[edge[0]], parts_list[edge[1]])
    
        return graph


    def continue_training(self, train_set, val_set, epochs=1):
        train_dataset = CustomGraphDataset(train_set)
        val_dataset = CustomGraphDataset(val_set)
        for t in range(epochs):
            self.train_and_validate(train_dataset, val_dataset)

    @classmethod
    def train_new_instance(cls, config_override: dict = None, checkpoint_dir=None, data_dir=None):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        # Load instances:
        dataset_retriever = DatasetRetriever.instance(override_path=data_dir)
        train_set = dataset_retriever.get_training_graphs()
        val_set = dataset_retriever.get_validation_graphs()

        new_instance = cls(config_override)
        train_dataset = CustomGraphDataset(train_set)
        val_dataset = CustomGraphDataset(val_set)

        mlflow.log_params(new_instance.get_meta_params())

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            new_instance._model.load_state_dict(model_state)
            new_instance._optimizer.load_state_dict(optimizer_state)

        print("Starting training...")

        # Start with a loss greater then any expected one to enable comparisons
        prev_normalized_val_loss = 10**6
        normalized_val_loss = 10**6 - 1
        # As many epochs, as specified, or as long as the loss on validation data decreases
        while (((meta_parameters.LEARNING_EPOCHS == 0)  
                    and normalized_val_loss < prev_normalized_val_loss) or 
                new_instance.epoch <= meta_parameters.LEARNING_EPOCHS):
            prev_normalized_val_loss = normalized_val_loss
            normalized_val_loss = new_instance.train_and_validate(train_dataset, val_dataset)
            print("Calculating edge accuracy on evaluation data:")
            edge_acc_validation = evaluate_edge_accuracy(new_instance, dataset_retriever.get_validation_graphs())
            print(f"Evaluation edge accuracy score on the evaluation dataset: {edge_acc_validation}")

            with tune.checkpoint_dir(new_instance.epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((new_instance._model.state_dict(), new_instance._optimizer.state_dict()), path)

            tune.report(loss=normalized_val_loss, accuracy=edge_acc_validation)

        mlflow.log_metric("training_epochs", new_instance.epoch - 1)
        return new_instance


    def train_and_validate(self, train_dataset, val_dataset) -> float:
        print(f"Epoch {self.epoch}")
        epoch_loss = self._train(train_dataset)

        print(f"Epoch {self.epoch}: Mean loss: {epoch_loss}")
        # loss on validation set
        loss = 0
        self._model.eval()
        for features, labels in val_dataset:
            features, labels = features.to(device), labels.to(device)
            curr_loss = self._loss(self._model(features), labels)
            loss += curr_loss
        
        if INTERMEDIATE_MODEL_STORE_PATH is not None:
            self.store_model(INTERMEDIATE_MODEL_STORE_PATH)

        normalized_val_loss = loss / (len(val_dataset))
        mlflow.log_metric("val_loss", normalized_val_loss,
                            (self.epoch) * len(train_dataset))
        print(f"Validation loss: {normalized_val_loss}")
        self.epoch += 1
        return normalized_val_loss

    def _train(self, dataset: CustomGraphDataset, batchsize=40):
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
        loss = 0
        
        progress_bar = tqdm(dataset) # Wraps progress bar around an interable 
        self._optimizer.zero_grad()
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            self._optimizer.zero_grad()

            preds = self._model(features)
            loss += self._loss(preds, labels)


            
            # print(loss.item())
            if self.step % batchsize == 0:
                loss.backward()
                self._optimizer.step()
                progress_bar.set_description(str(loss.item()/batchsize))
                progress_bar.update()
                mlflow.log_metric("train_loss",str(loss.item()/ batchsize), self.step)
                train_losses.append(loss.item()/batchsize)
                
                loss = 0 
                self._optimizer.zero_grad()


            self.step += 1 
            

        return sum(train_losses) / len(train_losses)

    def log_pytorch_models_to_mlflow(self):
        """
        Logs the model or models to mlflow
        """
        # Log the model to mlflow
        mlflow.pytorch.log_model(
            self._model,
            self.get_name()
        )

def edge_selection_strategy(preds, num_nodes):
    """
        for a flatt array of predictions, returns a list of edges (E, 2)
        Select n highest scoring edges 
    """
    selected_edges_in_flat_array = torch.topk(preds, num_nodes - 1).indices.to(device)
    all_edge_indices = torch.triu_indices(num_nodes, num_nodes, 1)
    transposed = all_edge_indices.t().to(device)

    selected_edges = transposed[selected_edges_in_flat_array]
    return selected_edges
