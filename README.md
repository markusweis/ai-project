# ai-lecture-project

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`conda create -n ai-project python=3.8`

2. Install the dependencies:  
`conda install -c conda-forge --file requirements.txt` 


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.

## MLflow
Start Mlflow UI with ```mlflow ui```.
It should be running on  http://localhost:5000.

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