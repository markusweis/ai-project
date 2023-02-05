from datetime import datetime
import numpy as np
from typing import  Set
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from tqdm import tqdm

from graph import Graph
from part import Part
from prediction_models.base_neural_network.base_graph_dataset import BaseGraphDataset
from prediction_models.base_neural_network.base_neural_network_model_definition import BaseNeuralNetworkModelDefinition
from prediction_models.base_prediction_model import BasePredictionModel
import mlflow
from prediction_models.base_neural_network import meta_parameters


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

BATCH_SIZE = 64

class NeuralNetworkPredictionModel(BasePredictionModel):
    """
    Fully connected neural network predicting adjacency matrizes
    """
    def __init__(self):
        self.model: BaseNeuralNetworkModelDefinition = BaseNeuralNetworkModelDefinition().to(device)
        self._loss_fn = nn.SmoothL1Loss()
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=meta_parameters.LEARNING_RATE)
        print("Using model:")
        print(self.model)

    @classmethod
    def get_name(self) -> str:
        """
        :return: Name of the model (used as model and run name)
        """
        return "base_neural_network_model"

    @classmethod
    def get_meta_params(self) -> dict:
        """
        :return: Dict containing all used meta parameters
        """
        return {
            "MAX_NUMBER_OF_PARTS_PER_GRAPH": meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH,
            "NUM_HIDDEN_LAYERS": meta_parameters.NUM_HIDDEN_LAYERS,
            "HIDDEN_LAYERS_SIZE": meta_parameters.HIDDEN_LAYERS_SIZE,
            "LEARNING_RATE": meta_parameters.LEARNING_RATE, 
            "UNUSED_NODES_PADDING_VALUE": meta_parameters.UNUSED_NODES_PADDING_VALUE,
        }

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
        missing_node_count = meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH - len(parts_list)
        if missing_node_count > 0:
            parts_tensor = pad(parts_tensor, (0, 0, 0, missing_node_count), "constant", meta_parameters.UNUSED_NODES_PADDING_VALUE)


        self.model.eval()
        with torch.no_grad():
            X = parts_tensor.to(device)
            pred = self.model(X)

        pred_one_dim = pred[0]

        return self.get_graph_from_nonredundand_connections_array_prediction(pred_one_dim, parts_list)

    def get_graph_from_nonredundand_connections_array_prediction(self, nonredundand_connections_array_prediction, parts_list) -> Graph:   
        # Instead of a threshold or only keeping the n-1 best scoring edges, make sure that every
        # node is connected and there are no cycles
        
        # Extract all non-padding edges:
        edges = list()
        padded_parts_len=meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH
        i = 0
        # Iterate through rows:
        for row in range(padded_parts_len - 1):
            if row >= len(parts_list):
                # In padding area
                for col in range(row + 1, padded_parts_len):
                    i += 1
                continue
            # Iterate through columns
            for col in range(row + 1, padded_parts_len):
                if col >= len(parts_list):
                    # In padding area
                    pass
                else:
                    # Add edge to both connected nodes:
                   edges.append((nonredundand_connections_array_prediction[i], row, col))
                i += 1

        # Sort to accept the highest predicted edges
        edges.sort(reverse=True)

        accepted_edges = list()
        connected_nodes = set()
        graph: Graph = Graph(datetime.now())
        
        for edge in edges:
            if edge[1] not in connected_nodes or edge[2] not in connected_nodes:
                accepted_edges.append(edge)
                connected_nodes.add(edge[1])
                connected_nodes.add(edge[2])
                graph.add_undirected_edge(parts_list[edge[1]], parts_list[edge[2]])

            # Quit if desired amount of edges reached
            if len(accepted_edges) == len(parts_list) - 1:
                break

        if graph.is_cyclic():
            print("A graph with a cycle was predicted!:")
            print(graph)

        if len(connected_nodes) < len(parts_list):
            print("Not all parts connected in graph:")
            print(graph)

        return graph

        # TODO
        # Stattdessen erst allerbeste
        # -> while statt for und immer popen, wenn verwendet
        # Nächstbeste, die anbaut an bisherigen knoten



        # Accept the best edge per part
        # -> This way, all parts get connected
        # -> The max amount of edges is then n (instead of the desired n-1)
        # -> Normally, however, two nodes should have the same edge as best prediction
        #       -> then, the desired n-1 is reached and no cycles are predicted
        # -> In rare cases with suboptimal training, one single cycle could be predicted

        graph: Graph = Graph(datetime.now())

        for i, edges in edges_per_part.items():
            best_pred_value = max([edge[0]for edge in edges])
            best_edge = [edge for edge in edges if edge[0] == best_pred_value][0]
            graph.add_undirected_edge(parts_list[best_edge[1]], parts_list[best_edge[2]])


        if graph.is_cyclic():
            print("A graph with a cycle was predicted!:")
            print(graph)

        return graph

        #######################################

        # Remove the padding values
        padded_parts_len=meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH
        i = 0
        # Iterate through rows:
        for row in range(padded_parts_len - 1):
            if row >= len(parts_list):
                # In padding area
                for col in range(row + 1, padded_parts_len):
                    pred_one_dim[i] = meta_parameters.UNUSED_NODES_PADDING_VALUE
                    i += 1
                continue
            # Iterate through columns
            for col in range(row + 1, padded_parts_len):
                if col >= len(parts_list):
                    # In padding area
                    pred_one_dim[i] = meta_parameters.UNUSED_NODES_PADDING_VALUE
                i += 1

        # Instead of a threshold, only keep the n-1 best scoring edges (with n being the number of nodes)
        max_edges = len(parts) - 1
        
        # begin with zeros
        final_reduzed_connections_array = torch.zeros_like(pred_one_dim)

        # Extract the n-1 largest values (flatten and unflatten the tensor)
        best_values, best_value_indices = torch.topk(pred_one_dim, max_edges)

        # Insert the values into final_pred
        for index in best_value_indices:
            final_reduzed_connections_array[index] = 1

        # Create the graph from the 
        graph = Graph.from_nonredundand_connections_array(
            part_order=parts_list, 
            reduzed_connections_array=final_reduzed_connections_array,
            pad_to_node_count=meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH
        )

        return graph

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        loaded_instance = cls()
        loaded_instance.model.load_state_dict(torch.load(file_path))
        return loaded_instance

    def log_pytorch_models_to_mlflow(self):
        """
        Logs the model or models to mlflow
        """
        # Log the model to mlflow
        mlflow.pytorch.log_model(
            self.model,
            self.get_name()
        )
    
    @classmethod
    def train_new_instance(cls, train_set: np.ndarray, val_set: np.ndarray):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        new_instance = cls()
        print("Loading training data...")
        train_dataset = BaseGraphDataset(train_set)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = BaseGraphDataset(val_set)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        
        print("Starting training...")
        mlflow.log_param("epochs", meta_parameters.LEARNING_EPOCHS)
        for t in range(meta_parameters.LEARNING_EPOCHS):
            print(f"Epoch {t+1}")
            new_instance._train(dataloader=train_dataloader)

            # loss on validation set
            loss = 0
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = new_instance.model(X)
                loss += new_instance._loss_fn(pred, y)
            normalized_val_loss = loss / (len(val_set) / BATCH_SIZE) # is the normlaization correct? 
            mlflow.log_metric("val_loss", normalized_val_loss, (t + 1) * len(train_set) )
            print(f"Validation loss: {normalized_val_loss}")    
        return new_instance


    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        torch.save(self.model.state_dict(), file_path)

    def _train(self, dataloader):
        self.model.train()
        progress_bar = tqdm(dataloader) # Wraps progress bar around an interable 
        for (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss = self._loss_fn(pred, y)
            mlflow.log_metric("train_loss", loss)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            progress_bar.set_description(str(loss.item()))
