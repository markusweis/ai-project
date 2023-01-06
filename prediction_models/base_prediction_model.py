from abc import ABC, abstractmethod
from typing import List, Set
from graph import Graph
from part import Part

class MyPredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @abstractmethod
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        pass
    
    @abstractmethod
    def train_and_store(self, train_graphs: List[Graph], file_path: str):
        """
        This method trains the prediction model with the given graphs 
        and stores it to a file (needed for evaluating your model on the test set).
        :param train_graphs: List of graphs to train with        
        :param file_path: path to file
        """
        pass