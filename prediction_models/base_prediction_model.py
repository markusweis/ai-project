from abc import ABC, abstractmethod
from typing import List, Set
from graph import Graph
from part import Part

class BasePredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @classmethod
    @abstractmethod
    def get_name(self) -> str:
        """
        :return: Name of the model (used as model and run name)
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_meta_params(self) -> dict:
        """
        :return: Dict containing all used meta parameters
        """
        pass

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
    
    @classmethod
    @abstractmethod
    def train_new_instance(cls, train_graphs: List[Graph]):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        pass

    @abstractmethod
    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        pass