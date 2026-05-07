#!/usr/bin/env python3
import argparse, os, sys
from typing import cast

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)
DSET_DIR: str = os.path.join(ROOT_DIR, "dataset")


class PyGAsTorchDataset(TorchDataset[Data]):
    """Typed `TorchDataset` wrapper over a PyG `TUDataset`."""

    def __init__(self, pyg_dataset: TUDataset) -> None:
        """Initialize with the underlying PyG dataset."""
        self.pyg_dataset: TUDataset = pyg_dataset

    def __len__(self) -> int:
        """Return the number of graphs."""
        return len(self.pyg_dataset)

    def __getitem__(self, idx: int) -> Data:
        """Return one graph sample as a `torch_geometric.data.Data` object."""
        return cast(Data, self.pyg_dataset[idx])


def load_tudataset_as_torch_dataset(
    name: str,
    root: str = DSET_DIR,
    use_node_attr: bool = False,
    use_edge_attr: bool = False,
) -> tuple[PyGAsTorchDataset, TUDataset]:
    """
    Load `TUDataset` and expose it as a typed Torch dataset of PyG `Data` objects.

    Args:
        name: The name of the TUDataset to load, such as "NCI1" or "NCI109" (required)
        root: The root directory to load the dataset from (default: dataset/)
        use_node_attr: Whether to use node attributes in the dataset (default: False)
        use_edge_attr: Whether to use edge attributes in the dataset (default: False)
    
    Returns:
        Tuple of `(torch_dataset, pyg_dataset)`.
    """
    pyg_dataset = TUDataset(
        root=root,
        name=name,
        use_node_attr=use_node_attr,
        use_edge_attr=use_edge_attr,
    )
    return PyGAsTorchDataset(pyg_dataset), pyg_dataset


def save_dataset_as_pt(dataset: PyGAsTorchDataset, out_path: str) -> list[Data]:
    """
    Save a `PyGAsTorchDataset` as a materialized `.pt` file of `list[Data]`.

    Args:
        dataset: The PyGAsTorchDataset to save
        out_path: The output path to save the .pt file to

    Returns:
        Materialized list of graph `Data` objects.
    """
    data_list: list[Data] = [dataset[i] for i in range(len(dataset))]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data_list, out_path)
    return data_list


def main() -> None:
    """CLI entrypoint for quick TU dataset inspection and optional serialization."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=DSET_DIR, help='dataset root folder')
    parser.add_argument('--dataset', required=True, help="dataset name, e.g., 'NCI1', 'NCI109'")
    parser.add_argument('--use-node-attr', action='store_true', help='Use node attributes (TUDataset backend only)')
    parser.add_argument('--use-edge-attr', action='store_true', help='Use edge attributes (TUDataset backend only)')
    parser.add_argument('--save-pt', default='', help='Optional output .pt path to save list[Data]')
    args = parser.parse_args()

    torch_dataset, _ = load_tudataset_as_torch_dataset(
        name=args.dataset,
        root=args.root,
        use_node_attr=args.use_node_attr,
        use_edge_attr=args.use_edge_attr,
    )

    if args.save_pt:
        save_dataset_as_pt(torch_dataset, args.save_pt)

    print(f'Loaded {len(torch_dataset)} graphs from {args.dataset}')
    print(f'Type of one sample: {type(torch_dataset[0]).__name__}')


if __name__ == '__main__':
    main()
