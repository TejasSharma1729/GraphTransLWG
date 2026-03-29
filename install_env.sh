#!/bin/bash
# Run: source install_env.sh

conda create -n graph-aug python=3.10
conda activate graph-aug
pip install -r requirements.txt
pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install ipykernel pynvim