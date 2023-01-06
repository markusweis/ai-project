from enum import Enum

from prediction_models.neural_network_prediction_model import NeuralNetworkPredictionModel
from prediction_models.straight_line_pseudo_prediction_model import StraightLinePseudoPredictionModel


class PredictionModels(Enum):
    NEURAL_NETWORK_PREDICTION_MODEL = "NEURAL_NETWORK_PREDICTION_MODEL"
    STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL = "STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL"

def get_model_class(model_type: str):
    if model_type == PredictionModels.NEURAL_NETWORK_PREDICTION_MODEL.value:
        return NeuralNetworkPredictionModel
    elif model_type == PredictionModels.STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL.value:
        return StraightLinePseudoPredictionModel
    