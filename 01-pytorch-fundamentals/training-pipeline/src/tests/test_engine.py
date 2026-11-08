import pytest
import torch
from torch import nn

from dlsl.engine import NonFiniteLossError, train_one_epoch


class FakeDetector(nn.Module):
    """A tiny model with the same loss-dictionary interface."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, targets):
        inputs = torch.stack(images)
        expected = torch.stack([
            target["expected"]
            for target in targets
        ])

        predictions = self.weight * inputs

        return {
            "loss_regression": (
                (predictions - expected) ** 2
            ).mean()
        }


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, kind, step, **values):
        self.records.append({
            "kind": kind,
            "step": step,
            **values,
        })


def config(accum_steps=1):
    return {
        "train": {
            "amp": False,
            "accum_steps": accum_steps,
            "clip_grad_norm": None,
            "log_every": 1,
            "checkpoint_every": 100,
        }
    }


def make_loader():
    return [
        (
            [torch.tensor([1.0])],
            [{"expected": torch.tensor([2.0])}],
        )
        for _ in range(5)
    ]


def test_loss_decreases():
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = RecordingLogger()

    step = train_one_epoch(
        model=model,
        loader=make_loader(),
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device="cpu",
        epoch=0,
        cfg=config(),
        logger=logger,
    )

    losses = [
        record["loss"]
        for record in logger.records
        if record["kind"] == "step"
    ]

    assert step == 5
    assert losses[-1] < losses[0]


def test_accumulation_includes_final_partial_group():
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    step = train_one_epoch(
        model=model,
        loader=make_loader(),
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device="cpu",
        epoch=0,
        cfg=config(accum_steps=2),
        logger=RecordingLogger(),
    )

    # Five batches form groups of 2, 2, and 1.
    assert step == 3


def test_nonfinite_loss_raises():
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loader = [
        (
            [torch.tensor([float("nan")])],
            [{"expected": torch.tensor([2.0])}],
        )
    ]

    with pytest.raises(NonFiniteLossError, match="global step 0"):
        train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device="cpu",
            epoch=0,
            cfg=config(),
            logger=RecordingLogger(),
        )