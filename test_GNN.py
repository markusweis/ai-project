from dataset_retriever import DatasetRetriever
import lovely_tensors as lt

from prediction_models.gnn.gnn_prediction_model import GNNPredictionModel

lt.monkey_patch()

# Load train data
dataset_retriever = DatasetRetriever.instance()
random_graph = dataset_retriever.get_random_graph()
gnn = GNNPredictionModel()
gnn.predict_graph(random_graph.get_parts())

