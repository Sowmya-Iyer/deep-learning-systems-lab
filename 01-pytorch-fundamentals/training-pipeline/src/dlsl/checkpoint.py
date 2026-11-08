"""Save and restore training state."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def _unwrap_model(model):
    """Get the original model beneath DDP or torch.compile wrappers."""
    while True:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def _rng_state():
    """Capture the current random-number generator states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(state):
    """Restore random-number generator states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_states = state["cuda"]

    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count differs from the saved checkpoint; "
                "use restore_rng=False for a non-exact resume"
            )

        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    step,
    best_metric,
    config,
):
    """Save a checkpoint, replacing the destination only after writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config,
        "rng": _rng_state(),
    }

    temporary_path = path.with_name(path.name + ".tmp")

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_rng=True,
):
    """Restore a trusted checkpoint produced by this pipeline."""
    # NumPy/Python RNG state requires full deserialization.
    # Only load trusted checkpoints written by this pipeline.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # Check requested state exists before restoring objects.
    for name, obj in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if obj is not None and checkpoint[name] is None:
            raise ValueError(
                f"Checkpoint has no {name} state to restore"
            )

    _unwrap_model(model).load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    # Restore randomness last: constructing fresh objects may consume RNG.
    if restore_rng:
        _restore_rng(checkpoint["rng"])

    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "best_metric": checkpoint["best_metric"],
        "config": checkpoint["config"],
    }


def find_last(run_dir):
    """Find last.pt, ignoring incomplete temporary files."""
    path = Path(run_dir) / "last.pt"
    return path if path.is_file() else None