#!/usr/bin/env python3
"""
Entry point for Table 4 ablation on ogbg-code2.

Three rows, three commands
--------------------------
Row 1  GNN-only baseline (train + save GNN checkpoint):
    python code2/table4/main.py --row gnn_only --save_path /path/to/checkpoints/

Row 2  GraphTrans with pre-trained + frozen GNN:
    python code2/table4/main.py --row frozen_gnn \\
        --pretrained /path/to/checkpoints/gnn_only_run0.pt \\
        --save_path  /path/to/checkpoints/

Row 3  GraphTrans with pre-trained + fine-tuned GNN:
    python code2/table4/main.py --row finetune_gnn \\
        --pretrained /path/to/checkpoints/gnn_only_run0.pt \\
        --save_path  /path/to/checkpoints/

All rows use paper settings by default (30 epochs, batch 16, lr 1e-4, 5 runs).
Override with --num_epochs, --batch_size, --learning_rate, --runs.
"""

import argparse, os, sys
from dataclasses import asdict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # GraphTransLWG/
sys.path.insert(0, _ROOT)

from data_utils.code2_tokenization import build_code2_dataset          # noqa: E402
from code2_gnn_initialization_ablation.train import run_experiment          # noqa: E402
from code2_gnn_initialization_ablation.configs import (                     # noqa: E402
    get_gnn_only_config,
    get_frozen_gnn_config,
    get_finetune_gnn_config,
)


def _device():
    import torch.cuda as _c, torch.mps as _m
    return torch.device("cuda" if _c.is_available() else "mps" if _m.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Table 4 ablation on ogbg-code2")
    parser.add_argument(
        "--row", required=True,
        choices=["gnn_only", "frozen_gnn", "finetune_gnn"],
        help="Which Table 4 row to run.",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="Path to a gnn_only checkpoint (.pt). Required for frozen_gnn / finetune_gnn.",
    )
    parser.add_argument("--save_path", default="checkpoints_table4", help="Directory for saving checkpoints.")
    parser.add_argument("--num_epochs",    type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--runs",          type=int,   default=5)
    parser.add_argument("--max_graphs",    type=int,   default=None,
                        help="Subsample dataset to this many total graphs (e.g. 4000).")
    args = parser.parse_args()

    if args.row in ("frozen_gnn", "finetune_gnn") and args.pretrained is None:
        parser.error(f"--pretrained is required for --row {args.row}")

    device = _device()
    print(f"Device: {device}", flush=True)

    # ── load dataset (shared across all rows) ────────────────────────────────
    (
        dataset, train_idx, val_idx, test_idx,
        vocab2idx, idx2vocab,
        num_nodetypes, num_nodeattributes,
    ) = build_code2_dataset(max_graphs=args.max_graphs)

    dataset_info = dict(
        dataset           = dataset,
        train_idx         = train_idx,
        val_idx           = val_idx,
        test_idx          = test_idx,
        vocab2idx         = vocab2idx,
        idx2vocab         = idx2vocab,
        num_nodetypes     = num_nodetypes,
        num_nodeattributes= num_nodeattributes,
        num_vocab         = len(vocab2idx),
        device            = device,
        dtype             = torch.float32,
    )

    # ── build kwargs for run_experiment ──────────────────────────────────────
    common = dict(
        num_epochs    = args.num_epochs,
        batch_size    = args.batch_size,
        runs          = args.runs,
        lr            = args.learning_rate,
    )

    if args.row == "gnn_only":
        print("\n── Row 1: GNN-only baseline ──────────────────────────────────", flush=True)
        kw = get_gnn_only_config(dataset_info, **common)

    elif args.row == "frozen_gnn":
        print("\n── Row 2: GraphTrans – pre-trained + frozen GNN ──────────────", flush=True)
        kw = get_frozen_gnn_config(dataset_info, args.pretrained, **common)

    else:  # finetune_gnn
        print("\n── Row 3: GraphTrans – pre-trained + fine-tuned GNN ──────────", flush=True)
        kw = get_finetune_gnn_config(dataset_info, args.pretrained, **common)

    # ── run ──────────────────────────────────────────────────────────────────
    experiment = run_experiment(**kw)

    # ── save checkpoints ─────────────────────────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    for run_id, result in enumerate(experiment.results):
        fname = os.path.join(args.save_path, f"{args.row}_run{run_id}.pt")
        torch.save(
            {
                "row":              args.row,
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
        print(f"Saved {fname}", flush=True)

    # ── final summary ─────────────────────────────────────────────────────────
    print(
        f"\n{'='*60}\n"
        f"Row: {args.row}\n"
        f"Valid F1 : {experiment.mean_val_metric:.4f} ± {experiment.std_val_metric:.4f}\n"
        f"Test  F1 : {experiment.mean_test_metric:.4f} ± {experiment.std_test_metric:.4f}\n"
        f"{'='*60}",
        flush=True,
    )


if __name__ == "__main__":
    main()
