#!/usr/bin/env python3
"""
Training loop for Table 4 — supports frozen-GNN mode.

Identical to models/train.py except:
  - Optimizer only includes parameters with requires_grad=True
    (so frozen GNN params are naturally excluded).
  - Exports run_experiment() for multi-run use.
"""

import copy, random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses (mirrors models/train.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    model: Module
    best_val_metric: float
    test_metric_at_best_val: float
    final_test_metric: float
    seed: int


@dataclass
class ExperimentResult:
    results: List[RunResult]
    mean_val_metric: float
    std_val_metric: float
    mean_test_metric: float
    std_test_metric: float


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_batches(indices: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def _evaluate(
    model: Module,
    device: torch.device,
    dataset: TorchDataset,
    indices: List[int],
    batch_size: int,
    out_mapping_fn: Callable,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metric_fn: Callable[[Tensor, Tensor], float],
) -> Tuple[float, float]:
    model.eval()
    total_loss = total_metric = total_n = 0.0

    for batch_idx in _make_batches(indices, batch_size):
        samples: List[Data] = [dataset[i] for i in batch_idx]  # type: ignore
        batch   = [s.to(device=device) for s in samples]
        gt: Tensor = out_mapping_fn(samples).to(device=device)
        out: Tensor = model(batch)
        total_loss   += float(loss_fn(out, gt).item()) * len(samples)
        total_metric += metric_fn(out, gt) * len(samples)
        total_n      += len(samples)

    n = max(total_n, 1)
    return total_loss / n, total_metric / n


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(
    model: Module,
    device: torch.device,
    dataset: TorchDataset,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    out_mapping_fn: Callable,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metric_fn: Callable[[Tensor, Tensor], float],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_name: str | None,
    seed: int,
    run_id: int | None = None,
) -> RunResult:
    """Train one run; optimizer automatically skips frozen params."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Only optimize parameters that are not frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=learning_rate, weight_decay=weight_decay)

    if scheduler_name is None or scheduler_name.lower() == "none":
        sched = None
    elif scheduler_name.lower() == "cosine":
        sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    best_val  = float("-inf")
    best_test = 0.0
    best_state = copy.deepcopy(model.state_dict())
    shuffle_gen = torch.Generator().manual_seed(seed)

    pbar = tqdm(range(num_epochs), desc=f"{'Run '+str(run_id)+' ' if run_id is not None else ''}Training")
    for epoch in pbar:
        model.train()
        order = torch.randperm(len(train_idx), generator=shuffle_gen).tolist()
        shuffled = [train_idx[i] for i in order]
        epoch_loss = 0.0
        epoch_n    = 0

        for batch_idx in _make_batches(shuffled, batch_size):
            optimizer.zero_grad()
            samples: List[Data] = [dataset[i] for i in batch_idx]  # type: ignore
            batch   = [s.to(device=device) for s in samples]
            gt: Tensor = out_mapping_fn(samples).to(device=device)
            out: Tensor = model(batch)
            loss: Tensor = loss_fn(out, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(samples)
            epoch_n    += len(samples)

        train_loss = epoch_loss / max(epoch_n, 1)
        val_loss,  val_m  = _evaluate(model, device, dataset, val_idx,  batch_size, out_mapping_fn, loss_fn, metric_fn)
        test_loss, test_m = _evaluate(model, device, dataset, test_idx, batch_size, out_mapping_fn, loss_fn, metric_fn)

        if val_m > best_val:
            best_val   = val_m
            best_test  = test_m
            best_state = copy.deepcopy(model.state_dict())

        if sched is not None:
            sched.step()

        prefix = f"Run {run_id} " if run_id is not None else ""
        print(
            f"{prefix}Epoch {epoch+1}/{num_epochs}  "
            f"train={train_loss:.4f}  val={val_m:.4f}  test={test_m:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        pbar.set_postfix(val=val_m, test=test_m)

    model.load_state_dict(best_state)
    _, final_test = _evaluate(model, device, dataset, test_idx, batch_size, out_mapping_fn, loss_fn, metric_fn)

    return RunResult(
        model=model,
        best_val_metric=best_val,
        test_metric_at_best_val=best_test,
        final_test_metric=final_test,
        seed=seed,
    )


def run_experiment(
    model_factory: Callable[[], Module],
    device: torch.device,
    dataset_factory: Callable[[], TorchDataset],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    out_mapping_fn: Callable,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metric_fn: Callable[[Tensor, Tensor], float],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_name: str | None,
    base_seed: int,
    runs: int,
) -> ExperimentResult:
    """Run multiple seeds and aggregate results."""
    dataset = dataset_factory()
    results: List[RunResult] = []

    for run_id in range(runs):
        seed  = base_seed + run_id
        model = model_factory()
        result = train_one_run(
            model, device, dataset,
            train_idx, val_idx, test_idx,
            out_mapping_fn, loss_fn, metric_fn,
            num_epochs, batch_size, learning_rate, weight_decay,
            scheduler_name, seed, run_id=run_id,
        )
        results.append(result)
        print(
            f"Run {run_id}  seed={seed}  "
            f"best_val={result.best_val_metric:.4f}  "
            f"test@best_val={result.test_metric_at_best_val:.4f}",
            flush=True,
        )

    vals  = np.array([r.best_val_metric          for r in results])
    tests = np.array([r.test_metric_at_best_val  for r in results])
    print(
        f"\n{'='*60}\n"
        f"Val  F1: {vals.mean():.4f} ± {vals.std():.4f}\n"
        f"Test F1: {tests.mean():.4f} ± {tests.std():.4f}\n"
        f"{'='*60}",
        flush=True,
    )
    return ExperimentResult(
        results=results,
        mean_val_metric=float(vals.mean()),
        std_val_metric=float(vals.std()),
        mean_test_metric=float(tests.mean()),
        std_test_metric=float(tests.std()),
    )
