import random

import numpy as np
import pytest
import torch
from torch import nn

from dlsl.checkpoint import (
    find_last,
    load_checkpoint,
    save_checkpoint,
)
from dlsl.seed import seed_everything


def build_training_objects():
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(8, 1),
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
    )

    return model, optimizer, scheduler


def training_step(model, optimizer, scheduler):
    model.train()

    # Exercise all three CPU random generators.
    X = torch.randn(4, 2)
    offset = random.random() + float(np.random.rand())
    y = X.sum(dim=1, keepdim=True) + offset

    optimizer.zero_grad()
    loss = ((model(X) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def test_resume_matches_uninterrupted_training(tmp_path):
    seed_everything(42, deterministic=True)

    model, optimizer, scheduler = build_training_objects()

    for _ in range(3):
        training_step(model, optimizer, scheduler)

    path = tmp_path / "last.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=0,
        step=3,
        best_metric=0.25,
        config={"seed": 42},
    )

    expected_loss = training_step(model, optimizer, scheduler)
    expected_state = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
    }

    # Construction consumes randomness; loading must restore it afterward.
    resumed_model, resumed_optimizer, resumed_scheduler = (
        build_training_objects()
    )

    progress = load_checkpoint(
        path,
        resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )

    actual_loss = training_step(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
    )

    assert actual_loss == pytest.approx(expected_loss, abs=1e-7)

    for name, tensor in resumed_model.state_dict().items():
        torch.testing.assert_close(
            tensor,
            expected_state[name],
            rtol=0,
            atol=0,
        )

    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert progress["step"] == 3
    assert progress["best_metric"] == 0.25
    assert progress["config"] == {"seed": 42}


def test_find_last_ignores_temporary_file(tmp_path):
    (tmp_path / "last.pt.tmp").write_bytes(b"incomplete")

    assert find_last(tmp_path) is None

    final_path = tmp_path / "last.pt"
    final_path.write_bytes(b"existing checkpoint")

    assert find_last(tmp_path) == final_path
    assert final_path.read_bytes() == b"existing checkpoint"