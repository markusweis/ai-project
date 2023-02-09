# ai-lecture-project

This project is written with Python `3.8` based on pip (https://pypi.org/project/pip/).

## Getting started

The file 'requirements.txt' lists the required packages.

We recommend to use the integrated Development Container within Visual Studio Code (https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).

Alternatively, you can manually create a virtual environment:

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`python3 -m venv env`

2. Install the dependencies:  
`python3 -m pip install -r requirements.txt` 


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.

## MLflow
The created ML models and experiments are managed with MLflow.

**Important:** The actual models got too large for a github repository. Therefore, they are stored in a cloud instead. You can load them here (https://nextcloud.timo-peter.de/s/A8wT4KzwnJfHfnj). To load them via mlflow, download the contents to `./mlartifacts/0`. This link is read-only. To get a link with write-access, ask the maintainers.

Before creating a new model, start Mlflow UI with ```mlflow ui```.
It should be running on  http://localhost:5000.

## Project Structure and Entrypoints
Besides the software tests mentioned above, this project has two entrypoints. Start them from within Visual Studio Code with the run profiles integrated into this project, or execute `python3 ENTRYPOINT_FILE_NAME.py ARGUMENT_LIST`.

### training.py
Trains a new model by using the training-dataset. The model is logged to MLflow, which has to be started first (see above). Optionally, the model is also stored at a given path.

Parameters:
1. SELECTED_MODEL_TYPE: Type of model as string defined in prediction_models/prediction_models_enum.py
2. SELECTED_MODEL_PATH: Path to store the model at (.pth or .zip, according to the type of model). (optional)

If no parameters are given, the model type and path defined via constants inside the script are selected.

### evaluation.py
To evaluate a model with the test-dataset, you can execute evaluation.py. 

Parameters:
1. SELECTED_MODEL_TYPE: Type of model as string defined in prediction_models/prediction_models_enum.py
2. SELECTED_MODEL_PATH: Path to the stored model (.pth or .zip, according to the type of model). To use an older model managed within MLflow, use the download functionality from within the UI and store the model at an suitable location.

If no parameters are given, the model type and path defined via constants inside the script are selected.

E.g., `python3 evaluation.py NEURAL_NETWORK_PREDICTION_MODEL prediction_models/model_instances/BASE_DNN.pth`


## Experiments and Models

### History of Experiments and Models

Multiple models were designed and trained within this project. See here for a detailed documentation:

[./experiments_and_models.md](./experiments_and_models.md)

### Currently Best Model:

GNN with the following meta-parameters:

- NUM_GNN_LAYERS=1
- EMBDEDDING_FEATURES = 64
- NUM_FC_LAYERS=2
- FC_FEATURES = 256
- LEARNING_RATE=  0.001
- WEIGHT_DECAY= 0
- DROPOUT=0

Achieved accuracy on the validation set: 97.31
Achieved accuracy on the test set: 97.19

You can find the model in `prediction_models/model_instances/FINAL_GNN.pth` Alternatively, you can download this model [here](https://nextcloud.timo-peter.de/s/TLACy5HwLH2spde).
The MLflow instance can be found [here](http://127.0.0.1:5000/#/experiments/0/runs/278197c940204ea09d4148d6e32f7544).

Store the model locally and use evaluation.py as described above to use it on the test-set.

## Notation
N Number of Nodes
E Nuber of Edges 
B Batch size 
F Number of Features -> 2 (id, family id)
d Number of Dimensions in the embbeding 
