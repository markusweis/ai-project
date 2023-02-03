from typing import List, Set

from graph import Graph
from part import Part
from prediction_models.base_prediction_model import BasePredictionModel





class StraightLinePseudoPredictionModel(BasePredictionModel):
    """
    Pseudo prediction model that forms a straight line of all given nodes and returns it.
    """

    @classmethod
    def get_name(self) -> str:
        """
        :return: Name of the model (used as model and run name)
        """
        return "straigt_line_pseudo_prediction_model"
    
    @classmethod
    def get_meta_params(self) -> dict:
        """
        :return: Dict containing all used meta parameters
        """
        return {}
    
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        # Pseudo output:
        graph = Graph()
        parts_list = list(parts) 
        for i in range(len(parts_list) - 1):
            graph.add_undirected_edge(parts_list[i], parts_list[i+1])
    
        return graph

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
        """
        loaded_instance = cls()
        return loaded_instance
    
    @classmethod
    def train_new_instance(cls, train_graphs: List[Graph]):
        """
        This method trains the prediction model with the given graphs 
        :param train_graphs: List of graphs to train with        
        """
        new_instance = cls()
        return new_instance


    def store_model(self, file_path: str):
        """
        This method stores the model to a file 
        (needed for evaluating your model on the test set).
        :param file_path: path to file
        """
        pass

    def log_pytorch_models_to_mlflow(self):
        """
        Logs the model or models to mlflow
        (nothing to log because here, no actual model is trained)
        """
        pass

    def _train(self, dataloader):
        pass