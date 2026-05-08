#!/usr/bin/env python3
"""
Table 5 ablation: readout type comparison on NCI1 / NCI109.

Runs all four readout variants (or one specific variant) and prints
a compact summary table at the end.

Usage
-----
# Run all 4 variants on NCI1 (paper-matching settings):
python table5/main.py --dataset NCI1

# Run a single variant:
python table5/main.py --dataset NCI109 --readout cls_cat

# Quick smoke-test (2 epochs, 2 runs):
python table5/main.py --dataset NCI1 --num_epochs 2 --runs 2

Results are also saved as .pt checkpoints in --save_path.
"""

import os, sys, argparse
from copy import replace
from dataclasses import replace as dc_replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from data_utils.config_objects import DATASET_CONFIGS, TRAIN_CONFIGS  # noqa: E402
from models.train import run_graph_transformer_experiments             # noqa: E402
from table5.model import GraphTransModelAblation, READOUT_TYPES        # noqa: E402


def run_variant(
    dataset_name: str,
    readout_type: str,
    num_epochs: int,
    runs: int,
    batch_size: int | None,
    learning_rate: float | None,
    save_path: str,
):
    config     = DATASET_CONFIGS[dataset_name]
    train_cfg  = TRAIN_CONFIGS[dataset_name]

    # Override model_loader with ablation model
    overrides = {
        "model_loader": lambda rt=readout_type: GraphTransModelAblation(config, rt),
    }
    if num_epochs    is not None: overrides["num_epochs"]    = num_epochs
    if runs          is not None: overrides["runs"]          = runs
    if batch_size    is not None: overrides["batch_size"]    = batch_size
    if learning_rate is not None: overrides["learning_rate"] = learning_rate

    train_cfg = dc_replace(train_cfg, **overrides)

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  |  readout = {readout_type}")
    print(f"{'='*60}")

    experiment = run_graph_transformer_experiments(train_cfg)

    # Save checkpoint for every run
    os.makedirs(save_path, exist_ok=True)
    for run_id, result in enumerate(experiment.results):
        fname = os.path.join(save_path, f"{dataset_name}_{readout_type}_run{run_id}.pt")
        torch.save(
            {
                "dataset":          dataset_name,
                "readout_type":     readout_type,
                "run_id":           run_id,
                "seed":             result.seed,
                "model_state_dict": result.model.state_dict(),
                "metrics": {
                    "best_val_metric":        result.best_val_metric,
                    "test_metric_at_best_val": result.test_metric_at_best_val,
                    "final_test_metric":       result.final_test_metric,
                },
            },
            fname,
        )

    return experiment.mean_test_metric, np.std(
        [r.test_metric_at_best_val for r in experiment.results]
    )


def main():
    parser = argparse.ArgumentParser(description="Table 5: readout ablation on NCI1/NCI109")
    parser.add_argument("--dataset",       type=str,   default="NCI1",
                        choices=["NCI1", "NCI109"])
    parser.add_argument("--readout",       type=str,   default="all",
                        choices=list(READOUT_TYPES) + ["all"],
                        help="Which readout variant to run (default: all)")
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--runs",          type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_path",     type=str,   default="checkpoints_table5")
    args = parser.parse_args()

    variants = list(READOUT_TYPES) if args.readout == "all" else [args.readout]

    results = {}
    for rt in variants:
        mean, std = run_variant(
            dataset_name  = args.dataset,
            readout_type  = rt,
            num_epochs    = args.num_epochs,
            runs          = args.runs,
            batch_size    = args.batch_size,
            learning_rate = args.learning_rate,
            save_path     = args.save_path,
        )
        results[rt] = (mean, std)

    # ── summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Table 5 summary — {args.dataset}")
    print(f"{'='*60}")
    print(f"  {'Readout':<12}  {'Test Acc (mean±std)':>24}")
    print(f"  {'-'*38}")
    for rt, (mean, std) in results.items():
        marker = " ←" if rt == "cls" else ""
        print(f"  {rt:<12}  {mean*100:>8.2f} ± {std*100:.2f}%{marker}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
