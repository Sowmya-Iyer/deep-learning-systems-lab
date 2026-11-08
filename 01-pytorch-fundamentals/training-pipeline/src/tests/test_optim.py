import pytest
import torch
from torch import nn

from dlsl.optim import build_optimizer, build_scheduler


def make_config():
    return {
        "optim": {
            "name": "sgd",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
        "schedule": {
            "name": "cosine",
            "epochs": 2,
            "warmup_iters": 2,
            "warmup_factor": 0.1,
        },
    }


def advance(model, optimizer, scheduler):
    optimizer.zero_grad()

    loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()

    optimizer.step()
    scheduler.step()


@pytest.mark.parametrize("name", ["sgd", "adamw"])
def test_optimizer_type(name):
    cfg = make_config()
    cfg["optim"]["name"] = name

    optimizer = build_optimizer(nn.Linear(2, 1), cfg)

    expected_type = (
        torch.optim.SGD if name == "sgd" else torch.optim.AdamW
    )

    assert isinstance(optimizer, expected_type)


def test_frozen_parameters_are_excluded():
    model = nn.Linear(2, 1)
    model.bias.requires_grad_(False)

    optimizer = build_optimizer(model, make_config())

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert id(model.weight) in optimized_ids
    assert id(model.bias) not in optimized_ids


def test_warmup_and_cosine_schedule():
    cfg = make_config()
    model = nn.Linear(2, 1)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer,
        cfg,
        iters_per_epoch=5,
    )

    # Base LR 0.1 × warmup factor 0.1.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    advance(model, optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.055)

    advance(model, optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    # Total schedule: 2 epochs × 5 updates = 10 updates.
    for _ in range(8):
        advance(model, optimizer, scheduler)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_step_decay():
    cfg = make_config()
    cfg["schedule"].update({
        "name": "step",
        "milestones": [4, 7],
        "gamma": 0.1,
    })

    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, iters_per_epoch=5)

    for _ in range(4):
        advance(model, optimizer, scheduler)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    for _ in range(3):
        advance(model, optimizer, scheduler)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_warmup_cannot_consume_entire_schedule():
    cfg = make_config()
    cfg["schedule"]["warmup_iters"] = 10

    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model, cfg)

    with pytest.raises(ValueError, match="warmup_iters"):
        build_scheduler(optimizer, cfg, iters_per_epoch=5)