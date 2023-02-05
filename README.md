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

to be defined...

## Notation
N Number of Nodes
E Nuber of Edges 
B Batch size 
F Number of Features -> 2 (id, family id)
d Number of Dimensions in the embbeding 

if torch.geometric functions are every necessary, it is possible to convert like this: 
data = Data(x=parts_list_tensor, edge_index=edge_index)

Unklarheiten (Timo Fragen)
In dem Blog werden die Node Features nicht benutzt. Das Embedding Modell wird immer nur auf 
den Embeddings aufgerufen. Kann man das überhaupt so machen, wie ich das versuche? 
Batch size? 


Fragen Timo -> Markus:
- Wie kommst du auf die Layer-Zusammensetzung im Baseline neural_network_prediction_model? Mir kommt es etwas gering vor, dass du in einer Zwischenstufe nur 32 Verbindungen hast

