# Experiments and Models

## 0. Pseudo-Model: Linear Concatenation
This is a simple solution without any actual Machine Learning. It is supposed to serve as a baseline.
The given parts are simply concatenated in a straigt line to form a graph.

**SELECTED_MODEL_TYPE:** STRAIGHT_LINE_PSEUDO_PREDICTION_MODEL

**Edge Accuracy – Evaluation dataset:** 69.84

**Edge Accuracy – Test dataset:** 70.35


## 1. Fully-connected Neural Network
The following experiments implement a fully-connected neural network in different variations considering both meta-parameters and the handling of the input- and output-data (Padding, Predicted adjacency matrix to actual graph, ...).

In common, the input for the neural network is a "parts_tensor" consisting of each one array of part_ids and family_ids. 
Since a basic fully-connected neural network expects inputs of equal length, the tensor is padded to a fixed value (we selected to support 30 Nodes).
In order to reduce the input and output space, we sort the parts given to the model according to their part_id, as well as the node order in the expected adjacency matrizes in the training step.

The direct output of the neural network is a predicted continuos-value adjacency matrix ("probability"-matrix). The process of creating the actual discrete adjacency matrix and therefore the graph, is dependend on the experiment described below:

**SELECTED_MODEL_TYPE:** NEURAL_NETWORK_PREDICTION_MODEL

### 1.1 Threshold-based Adjacency Matrix

In this early experiments, a simple threshold was defined to predict a discrete adjacency matrix from the "probability"-matrix.

As can be seen in the results, this does not work well due to too many edges being created.

#### 1.1.1 
An initial meta-parameter setup with 0.5 as threshold for the adjacency matrix discretization.

Main finding: Nearly all predicted graphs are fully connected. 

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/8fc83549563d4814af40dcb2c08a8a14)

[git commit](http://127.0.0.1:5000/#/experiments/0/runs/3bd13ad1e3534ceaad05ba7140c010ae)

**Edge Accuracy – Evaluation dataset:** 42.33

**Edge Accuracy – Test dataset:** 43.98

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 0.5

----------
Open ideas for further experiments (fully-connected):
- Only best n-1 edges
- Only best edge per node
- One-hot encoding
- Padding with -1 or 0
- Random instead of sorting

## 2. Graph-convolutional Neural Network


---------------------------

General further ideas:
- graph-attention?
- graph-transformer?
- Generate more training-data?