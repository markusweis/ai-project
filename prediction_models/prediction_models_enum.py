from enum import Enum

from prediction_models.neural_network_prediction_model import NeuralNetworkPredictionModel


class PredictionModels(Enum):
    NEURAL_NETWORK_PREDICTION_MODEL = "NEURAL_NETWORK_PREDICTION_MODEL"

def get_model_class(model_type: str):
    if model_type == PredictionModels.NEURAL_NETWORK_PREDICTION_MODEL.value:
        return NeuralNetworkPredictionModel
