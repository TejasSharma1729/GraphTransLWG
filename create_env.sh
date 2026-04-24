#!/bin/bash
set -e

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install Miniconda or Anaconda and ensure it is on your PATH." >&2
    exit 1
fi

# conda activate doesn't work in non-interactive shells without this
eval "$(conda shell.bash hook)"

echo "Checking for existing 'graphTrans' conda environment..."
if conda env list | grep -q "^graphTrans "; then
    echo "Environment 'graphTrans' already exists. Removing it..."
    conda deactivate 2>/dev/null || true
    conda env remove -n graphTrans -y
fi

echo "Creating conda environment 'graphTrans' with Python 3.10..."
conda create -n graphTrans python=3.10 -y
conda activate graphTrans

echo "Installing PyG and its dependencies using conda (pip breaks it) ..."
conda install pyg -c pyg -y

echo "Installing other project dependencies using pip..."
pip install -r requirements.txt

echo "Extracting and saving datasets..."
python data_utils/extract_datasets.py

echo "Environment 'graphTrans' created and dependencies installed successfully!"
echo "Run 'conda activate graphTrans' to work in this environment."
