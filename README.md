# GraphTransLWG: A Graph Trans Implementation.

An implementation of `GraphTrans` by Tejas Sharma, Eshaan Pareek, Matam Kushaal and Aditya Verma Reddy.

## Code Structure
** `main.py`: Global entry point to train or test using traind `GraphTransLWG` models.
** `create_env.sh`: Script for quick setup of the environment `graphTrans` with all dependencies.

### Directory `models`
Contains the code for the components of our GraphTrans model implementation:
** `batch_utils.py`: Utilities for extracting CLS mask, batch edge matrix and weighed attention matrix (based on distance factors)
** `gnn.py`: GNNLayer and GNN module, message-passign GELU-based GNN
** `attention.py`: Attention layer (with weighed per-graph attention mask)
** `mlp.py`: MLP (per-node computation), seperate for graph nodes and CLS nodes
** `transformer.py`: Transformer (layer and full), each layer combines a GNN, Attention and MLP
** `full_model.py`: Modular architecture that contains initial embedding of graph inputs (linear layer), CLS node initial embedding, final-unembedding (of CLS node) to extract output feature. Also contains helper classes for train config, and the model config.
** `train.py`: Utilities to train models, run multiple training epochs and record losses as well as other metrics with the trained model in helper objects.

### Directory `data_utils`
Contains the code to extract datasets and get the model and train configs for various datasets.
** `extract_datasets.py`: load and extract OGBG datasets: Molhiv, MolPCBA, Code2
** `tu_to_pyg.py`: load and extract the NCI1 and NCI109 graph datasets
** `config_objects.py`: the model and train configs for each of the 5 datasets supported
** `code2_tokenization.py`: utilities to interpret the code2 graph as ASTs (max. 20 node-depth) and the vocabulary of the output (the tokens) and return them with the dataset and the transform function.
** `code2_train_config.py`: utilities to craft the metric function, the loss function and  

### Directory `code2_gnn_initialization_ablation`
Contains code for GNN ablation tests (seperate training of GNN vs joint training).

### Directory `ncl_readout_ablation`
Contains code for ablation tests on what to use for unembedding (just `CLS` or the average of graph node embeddings or both, or the last node embedding per graph).

### Directory `report`
Contains the LaTeX and PDF format report

### Directory `deprecated_trainers`
Contains unused `BaselineTrainer` and `FlagTrainer` static classes and methods to register them.

### Directory `checkpoints`
Contains trained PyTorch models saved in `.pt` form (may not be in the repo)

### Directory `dataset`
Contains stored datasets for efficient reuse.