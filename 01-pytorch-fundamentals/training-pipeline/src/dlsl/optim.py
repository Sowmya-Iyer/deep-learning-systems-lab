"""Optimizer factories and learning-rate schedules."""

import math

import torch


def build_optimizer(model, cfg):
    """Build an optimizer using only trainable parameters."""
    options = cfg["optim"]

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError("Model has no trainable parameters")

    name = options["name"].lower()
    lr = options["lr"]
    weight_decay = options.get("weight_decay", 0.0)

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=options.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name!r}")

def build_scheduler(optimizer, cfg, iters_per_epoch):
    """Build a schedule stepped once per successful optimizer update.

    iters_per_epoch means optimizer updates per epoch,
    not necessarily the number of DataLoader batches.
    """
    options = cfg["schedule"]

    epochs = options["epochs"]
    warmup_iters = options.get("warmup_iters", 0)
    warmup_factor = options.get("warmup_factor", 0.001)

    if iters_per_epoch <= 0 or epochs <= 0:
        raise ValueError("iters_per_epoch and epochs must be positive")

    total_steps = epochs * iters_per_epoch

    if not 0 <= warmup_iters < total_steps:
        raise ValueError(
            "warmup_iters must be nonnegative and smaller than total steps"
        )

    if not 0 < warmup_factor <= 1:
        raise ValueError("warmup_factor must be in (0, 1]")

    name = options["name"].lower()

    if name not in {"cosine", "step"}:
        raise ValueError(f"Unknown scheduler: {name!r}")

    # Step-decay milestones are measured from the start of training,
    # in completed optimizer updates.
    milestones = sorted(options.get("milestones", []))
    gamma = options.get("gamma", 0.1)

    if name == "step":
        if any(
            type(step) is not int
            or not warmup_iters < step < total_steps
            for step in milestones
        ):
            raise ValueError(
                "Each milestone must be an integer after warmup "
                "and before the end of training"
            )

        if len(milestones) != len(set(milestones)):
            raise ValueError("Milestones must be unique")

        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")

    def lr_multiplier(step):
        # LambdaLR supplies a step index starting at zero.
        step = min(step, total_steps)

        if step < warmup_iters:
            progress = step / warmup_iters

            return (
                warmup_factor
                + progress * (1.0 - warmup_factor)
            )

        if name == "cosine":
            progress = (
                (step - warmup_iters)
                / (total_steps - warmup_iters)
            )

            return 0.5 * (1.0 + math.cos(math.pi * progress))

        decay_count = sum(
            step >= milestone
            for milestone in milestones
        )

        return gamma ** decay_count

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )