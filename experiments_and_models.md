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

Main finding: Since the output is standardized to a mean (μ) of 0 and standard deviation (σ) of 1, also values above 1.0 are predicted. This shows here, as the results get dramatically better.

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

This is kept as the best version of the threshold-based technique. Other meta-parameters are not optimized here, because we believe to perform better with another strategy implemented in 1.2.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/0e09cc8f920c44c79c91e99d88072aaa)

[git commit](https://github.com/markusweis/ai-project/tree/db38b53adf584cfe0f0c2be1077fd5995a7c211c)

**Edge Accuracy – Evaluation dataset:** 77.48

**Edge Accuracy – Test dataset:** 77.68

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- ADJACENCY_MATRIX_HIT_THRESHOLD = 1.25

### 1.2 Only Node-count - 1 Edges
In the previous experiments, often a wrong amount of edges were predicted, consequentially leading to suboptimal results.
An approach to overcome the issue of predicting the wrong amount of edges can be to only accept the node-count -1 edges with the highest prediction. This way, the 

The remainder of meta-parameters is kept the same for this experiment. 

Main finding: Now no graphs with too many edges are predicted, but still some graphs have too few edges and nodes not connected. This can be due to the adjacency matrix being symmetrical (undirected graph). The same edge can be predicted multiple times, leaving no one for other connections. Additionally, sometimes cycles are predicted instead of the desired tree-structures. This should be adressed in further experiments.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/93e5273f6a3b4207bfeaa7331e5af155)

[git commit](https://github.com/markusweis/ai-project/tree/5bf0a7125229cbe32486d0eebe6b81c8d60bbcb9)

**Edge Accuracy – Evaluation dataset:** 72.47

**Edge Accuracy – Test dataset:** 72.43

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1

### 1.3 Non-redundant Edge-output
Until here, the output of the model was a full adjacency matrix. Since the desired graph is undirected and has no self-edges, this is highly redundant.
This was changed here. Instead of a adjacency matrix, a "nonredundant_connections_array" is predicted, that is composed like follows:

Adjacency matrix indices:
``` 
           A | B | C | ?
        A  0   1   2   3 
        B  4   5   6   7
        C  8   9   10  11
        ?  12  13  14  15
``` 

Content indices of the reduced array:
[1, 2, 3, 6, 7, 11]

Otherwise, still the n-1 predicted edges with the highest scores were selected.

Main finding: Now the predicted graphs always have the correct amount of edges. Still, the edge accuracy is about the same as for 1.3 and even lower than for 1.1.5. The reason is that now, often cyclic graphs are predicted. In comparison to the previous experiment, this means two wrong edge-presences (the one closing the cycle is too much, whereas the correct one is missing). Before, often there was only one wrong edge-presence: A missing one. 

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/6da91574faab4976b755ebd02924caef)

[git commit](https://github.com/markusweis/ai-project/tree/a71859a3a904760834ca40124db55f26654da64d)

**Edge Accuracy – Evaluation dataset:** 71.82

**Edge Accuracy – Test dataset:** 72.48

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1


### 1.4 Best Edge Per Node
To solve the issue of having nodes without any edges – and therefore cycles at other places – a
strategy is to accept one (best) edge per node. 

Accept the best edge per part
- This way, all parts get connected
- The max amount of edges is then n (instead of the desired n-1)
- Normally, however, two nodes should have the same edge as best prediction
    - then, the desired n-1 is reached and no cycles are predicted
- In rare cases with suboptimal training, one single cycle could be predicted

Main finding: The edge accuracy is improved, but another issue occurs: While now, all parts are connected with at least one edge, this does not mean that they all are connected to a single graph. Instead, the same edges are considered best for multiple nodes, keeping a lower number of total edges
than the desired n-1.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/ba0cb07fa16b4d2db284a0960e5ac9c4)

[git commit](https://github.com/markusweis/ai-project/tree/17106337379bb4f1cfb687b96ff281bd319ec5f9)

**Edge Accuracy – Evaluation dataset:** 76.95

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1

### 1.5 Cycle-free Global Best Edges
Since the previous strategy to overcome the issue of unconnected graph outputs, another strategy was developed:

First, the global highest predicted edge is accepted as starting point.
Going from there, the next best edge is chosen, that is connected to the already known
parts of the graph and adds exactly one new node to the structure. Thereby, no cycles
can be created. This is repeated until n-1 edges are selected. By then, also every 
given part is connected into the single graph.

Main finding: The structure of the predicted graphs now is as desired: All nodes connected and cycle-free. The edge accuracy is still below the best previous results and needs improvement on other elements, like the input and meta-parameters. The reason for worse edge accuracy in comparison to e.g., 1.1.5 is again, that a wrong edge is considered worse than no edge.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/05763fdd7466471ea153f3eabaf065e7)

[git commit](https://github.com/markusweis/ai-project/tree/a83f8b14bee7b23ea7952552ac0ee530eec5b7da)

**Edge Accuracy – Evaluation dataset:** 75.06

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1

### 1.6 One-hot Encoding
The next experiment adresses potential performance-flaws related to the model input.
Until here, the part_id and family_id were directly used as input. This e.g., makes parts with IDs 3 and 4 more similar then IDs 3 and 800, although semantically, the latter might be more similar.

To solve this, a dual one-hot encoding is used in this experiment (for both the part_id and family_id).
As analyzed with `dataset_statistics.py`, the maximum part_id is 2270 and the maximum family_id is 100. The according ranges are therefore used for the encoding.

#### 1.6.1
Within this initial one-hot experiment, the remainder of meta-parameters was kept the same.

Main finding: The meta-parameters seem suboptimal, as the accuracy is worse than without the one-hot encoding.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/bb0ec35c01f3479bb6a932dffbe74849)

[git commit](https://github.com/markusweis/ai-project/tree/7cb8e4cc78aaa337604aaa04cb3a6e1c4a17798b)

**Edge Accuracy – Evaluation dataset:** 72.91

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = -1
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.2 Unused nodes padding value
The first meta-parameter to optimize was the expected output-value for the padding area. Multiple values were tested and are bundled here, because only one value changed.

Main finding: The UNUSED_NODES_PADDING_VALUE has a large impact on the resulting edge accuracy of the actual nodes. A value of 0.8 seems to be the optimum.

[MLflow experiment (best with parameter 0.8)](http://127.0.0.1:5000/#/experiments/0/runs/bb0ec35c01f3479bb6a932dffbe74849)

[git commit](https://github.com/markusweis/ai-project/tree/7cb8e4cc78aaa337604aaa04cb3a6e1c4a17798b)

**Edge Accuracy – Evaluation dataset (UNUSED_NODES_PADDING_VALUE=0.8):** 81.07

Other edge accuracies:
- UNUSED_NODES_PADDING_VALUE = 0 -> edge accuracy: 75.18
- UNUSED_NODES_PADDING_VALUE = 1.0 -> edge accuracy: 80.63
- UNUSED_NODES_PADDING_VALUE = 0.5 -> edge accuracy: 80.7
- UNUSED_NODES_PADDING_VALUE = 2 -> edge accuracy: 71.81
- UNUSED_NODES_PADDING_VALUE = 100 -> edge accuracy: 72.42
- UNUSED_NODES_PADDING_VALUE = -100 -> edge accuracy: 73.25
- UNUSED_NODES_PADDING_VALUE = 1.1 -> edge accuracy: 78.73
- UNUSED_NODES_PADDING_VALUE = 0.9 -> edge accuracy: 80.97
- UNUSED_NODES_PADDING_VALUE = 0.8 -> edge accuracy: 81.07
- UNUSED_NODES_PADDING_VALUE = 0.7 -> edge accuracy: 80.94
- UNUSED_NODES_PADDING_VALUE = 0.75 -> edge accuracy: 80.67
- UNUSED_NODES_PADDING_VALUE = 0.85 -> edge accuracy: 81.06

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE as mentioned above
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.3 Removed standardization
A standardization layer was previously included, that was introduced for the threshold solutions in 1.1.*. This was removed, as it is no longer required

Main finding: The loss function now gets way closer to 0. 

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/ff6937e4f6e742fbb9cb8b3ffb1af791)

[git commit](https://github.com/markusweis/ai-project/tree/19c0769ac79122078e731a2d463a4023047aaed2)

**Edge Accuracy – Evaluation dataset:** 81.05

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = 0.8
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.4 Mean Squared Error
Changed to the Mean Squared Error (MSE) loss function.

Main finding: MSE seems to perform a little better than L1Loss

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/173e07768efc426bb553fb7fc6c29cc5)

[git commit](https://github.com/markusweis/ai-project/tree/90dacd4b5526645d525f0e2a6b8568b2242e116a)

**Edge Accuracy – Evaluation dataset:** 81.15

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 512
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = 0.8
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.5 Larger hidden layer
In this experiment, a way larger hidden layer was used, that is closer to the size of the input dimensionality.

Main finding: Way more computation power required, without any benefit in resulting edge accuracy.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/b39ed3fc57514c3190c1e7db4980e97d)

[git commit](https://github.com/markusweis/ai-project/tree/dbb772b073abe7c97c8a100996eb285f5a134a60)

**Edge Accuracy – Evaluation dataset:** 81.17

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 1
- HIDDEN_LAYERS_SIZE = 5000
- LEARNING_RATE = 0.05
- LEARNING_EPOCHS = 5
- UNUSED_NODES_PADDING_VALUE = 0.8
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.6 More layers
In this experiment, 10 hidden layers were used instead of just one. The size per layer was reduced back to 435, which matches the output size.

Main finding: Increasing the complexity of the model does not increase the performance significantly. The loss also still decreases quite fast in the second training epoch.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/1845c2c4d3d74e2c934e97f70ca04be3)

[git commit](https://github.com/markusweis/ai-project/tree/818f09de24962f58ce3b38c428f5f690b7a92b3a)

**Edge Accuracy – Evaluation dataset:** 81.22

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 10
- HIDDEN_LAYERS_SIZE = 435  # Like the output size
- LEARNING_RATE = 0.1
- LEARNING_EPOCHS =  5 # was unbounded -> stopped when the loss on validation data did not increase anymore
- UNUSED_NODES_PADDING_VALUE = 0.8
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.7 Faster training
In this experiment, the learning rate was increased from 0.1 to 1.0

Main finding: Increasing the learning rate allows for an even much faster training with about the same result.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/fa48afc72b5e436aabb0015d94de192b)

[git commit](https://github.com/markusweis/ai-project/tree/818f09de24962f58ce3b38c428f5f690b7a92b3a)

**Edge Accuracy – Evaluation dataset:** 81.22

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 10
- HIDDEN_LAYERS_SIZE = 435  # Like the output size
- LEARNING_RATE = 1
- LEARNING_EPOCHS =  2 # was unbounded -> stopped when the loss on validation data did not increase anymore
- UNUSED_NODES_PADDING_VALUE = 0.8
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100

#### 1.6.8 Custom loss function
In this experiment, a new, custom loss function was introduced. Instead of calculating the loss based on the whole prediction including the padding space, this version ignores values in the padding and only focuses on the actual edges.

Main finding: Worse than without this additional focus -> Idea dropped.

[MLflow experiment](http://127.0.0.1:5000/#/experiments/0/runs/97c8cb3cf07c4a9dbf5b7523427e08eb)

[git commit](https://github.com/markusweis/ai-project/tree/00a5febb4f41e87b1a3acdc33ac7695f6eb6dbe2)

**Edge Accuracy – Evaluation dataset:** 72.62

**Meta-parameters:**
- MAX_NUMBER_OF_PARTS_PER_GRAPH = 30
- NUM_HIDDEN_LAYERS = 10
- HIDDEN_LAYERS_SIZE = 435  # Like the output size
- LEARNING_RATE = 0.2
- LEARNING_EPOCHS =  2 # was unbounded -> stopped when the loss on validation data did not increase anymore
- UNUSED_NODES_PADDING_VALUE = -1
- MAX_SUPPORTED_PART_ID = 2270
- MAX_SUPPORTED_FAMILY_ID = 100


## 2. Graph-convolutional Neural Network


---------------------------

General further ideas:
- graph-attention?
- graph-transformer?
- Generate more training-data?