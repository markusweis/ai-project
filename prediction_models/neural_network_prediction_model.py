from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
from enum import Enum

from graph import Graph
from node import Node
from part import Part
from prediction_models.base_prediction_model import MyPredictionModel


class NeuralNetworkPredictionModel(MyPredictionModel):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        pass

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        pass

    def train_and_store(self, train_graphs: List[Graph], file_path: str):
        """
        This method trains the prediction model with the given graphs 
        and stores it to a file (needed for evaluating your model on the test set).
        :param train_graphs: List of graphs to train with        
        :param file_path: path to file
        """
        pass