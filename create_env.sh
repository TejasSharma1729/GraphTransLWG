#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

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

echo "Upgrading pip and installing networking/HTTP dependencies to avoid PyCAres issues..."
python -m pip install --upgrade pip setuptools wheel
# Ensure compatible pycares/aiodns/aiohttp/fsspec are present to avoid runtime import errors
python -m pip install --upgrade pycares aiodns aiohttp fsspec || true

echo "Attempting to install PyG via conda (preferred). If this fails, we'll continue and let pip handle remaining deps."
if ! conda install pyg -c pyg -y; then
    echo "Warning: conda install pyg failed; continuing (pip install may install torch-geometric from wheels)."
fi

echo "Installing other project dependencies using pip..."
pip install -r requirements.txt

echo "Extracting and saving datasets..."
python data_utils/extract_datasets.py

echo "Preparing TU raw datasets (NCI1, NCI109) if missing..."
mkdir -p dataset/processed_tu
python data_utils/tu_to_pyg.py --dataset NCI1 --root dataset --save-pt dataset/processed_tu/NCI1.pt
python data_utils/tu_to_pyg.py --dataset NCI109 --root dataset --save-pt dataset/processed_tu/NCI109.pt

echo "Environment 'graphTrans' created and dependencies installed successfully!"
echo "Run 'conda activate graphTrans' to work in this environment."
