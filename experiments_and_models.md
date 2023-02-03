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

[git commit](https://github.com/markusweis/ai-project/tree/a125257b51e0fa9a53000f45e92a667e247893b6)

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

#### 1.1.2
A naive approach to fix the problem of 1.1.1 predicting nearly only fully connected graphs: Adjusting the threshold: First try: 0.5 -> 0.8

Main finding: A bit better, but still nearly all predicted graphs are fully connected -> Needs more increasing. 

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/07dbce5b6a9646c0adb5da96d663fdfe)

[git commit](https://github.com/markusweis/ai-project/tree/d78bab422b31ccb4e394ab34e1381eac9d4dd5d4)

**Edge Accuracy – Evaluation dataset:** 48.71

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 0.8

#### 1.1.3
A naive approach to fix the problem of 1.1.1 predicting nearly only fully connected graphs: Adjusting the threshold: Second try: 0.8 -> 1.0

Main finding: 

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/259e1b9a212f4684a7ef26891bfc59a2)

[git commit](https://github.com/markusweis/ai-project/tree/da8e3f09606a26abfec5822284ed752af397e51a)

**Edge Accuracy – Evaluation dataset:** 64.75

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 1.0

#### 1.1.4
A naive approach to fix the problem of 1.1.1 predicting nearly only fully connected graphs: Adjusting the threshold: Third try: Larger increase from 1.0 to 1.5. Expectation: worse than for 1.0. because too few edges are predicted.

Main finding: Many graphs without edges -> Too large threshold. But higher edge accuracy score of 79.64

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/999e81a8453b4c73b1dc325157905224)

[git commit](https://github.com/markusweis/ai-project/tree/c2978855aadc829ff4e83e87e3f6aaf93691e8ef)

**Edge Accuracy – Evaluation dataset:** 

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 1.5

#### 1.1.5
A naive approach to fix the problem of 1.1.1 predicting nearly only fully connected graphs: Adjusting the threshold: First try: 1.5 -> 1.25

Main finding: Results look way more like the targetted graphs. Some nodes are not connected. Some even are empty, without any edges. The result is subjectively better based on some visually compared samples, although the edge accuracy score is a little lower.

This is kept as the best version of the threshold-based technique.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/0e09cc8f920c44c79c91e99d88072aaa)

[git commit]()

**Edge Accuracy – Evaluation dataset:** 77.48

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 1.25


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