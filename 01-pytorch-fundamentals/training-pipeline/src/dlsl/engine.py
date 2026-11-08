"""Training and evaluation loops."""

import torch

from dlsl.runlog import Throughput, peak_memory_mb


class NonFiniteLossError(RuntimeError):
    """Raised when a loss contains NaN or infinity."""

    def __init__(self, step, losses):
        super().__init__(
            f"Non-finite loss at global step {step}: {losses}"
        )
        self.step = step
        self.losses = losses


def _move_target(target, device):
    """Move target tensors while preserving other metadata."""
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in target.items()
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    cfg,
    logger,
    checkpoint_fn=None,
    skip_batches=0,
    global_step=0,
):
    """Train one epoch and return the successful optimizer-update count.

    checkpoint_fn receives keyword arguments:
        epoch, step, batches_completed

    Resume is supported only at accumulation boundaries.
    """
    device = torch.device(device)
    options = cfg["train"]

    accum_steps = options.get("accum_steps", 1)
    log_every = options.get("log_every", 20)
    checkpoint_every = options.get("checkpoint_every", 500)
    clip_norm = options.get("clip_grad_norm")
    amp_enabled = options.get("amp", False) and device.type == "cuda"

    num_batches = len(loader)

    if accum_steps < 1:
        raise ValueError("accum_steps must be positive")

    if log_every < 1 or checkpoint_every < 1:
        raise ValueError("Logging/checkpoint intervals must be positive")

    if not 0 <= skip_batches <= num_batches:
        raise ValueError("skip_batches is outside the loader")

    if skip_batches != num_batches and skip_batches % accum_steps:
        raise ValueError(
            "Resume must start at an accumulation boundary"
        )

    if amp_enabled and (scaler is None or not scaler.is_enabled()):
        raise ValueError("CUDA AMP requires an enabled GradScaler")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    throughput = Throughput(device)
    peak_memory_mb(reset=True)

    skipped_updates = 0

    for batch_index, (images, targets) in enumerate(loader):
        if batch_index < skip_batches:
            continue

        images = [
            image.to(device, non_blocking=True)
            for image in images
        ]
        targets = [
            _move_target(target, device)
            for target in targets
        ]

        # The final accumulation group may contain fewer batches.
        group_start = (batch_index // accum_steps) * accum_steps
        group_size = min(accum_steps, num_batches - group_start)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss_dict = model(images, targets)

            if not loss_dict:
                raise ValueError("Model returned an empty loss dictionary")

            total_loss = sum(loss_dict.values())

            # Average across microbatches in this accumulation group.
            backward_loss = total_loss / group_size

        loss_values = {
            name: value.detach().item()
            for name, value in loss_dict.items()
        }

        if (
            not all(
                torch.isfinite(value.detach()).all().item()
                for value in loss_dict.values()
            )
            or not torch.isfinite(total_loss.detach()).item()
        ):
            optimizer.zero_grad(set_to_none=True)
            raise NonFiniteLossError(global_step, loss_values)

        if amp_enabled:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        throughput.update(len(images))

        group_finished = (
            (batch_index + 1) % accum_steps == 0
            or batch_index + 1 == num_batches
        )

        if not group_finished:
            continue

        if amp_enabled:
            scaler.unscale_(optimizer)

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_norm,
            )

        if amp_enabled:
            previous_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            # With the standard GradScaler, overflow decreases the scale
            # and causes the optimizer update to be skipped.
            update_succeeded = scaler.get_scale() >= previous_scale
        else:
            optimizer.step()
            update_succeeded = True

        optimizer.zero_grad(set_to_none=True)

        if not update_succeeded:
            skipped_updates += 1
            continue

        global_step += 1

        if scheduler is not None:
            scheduler.step()

        if global_step % log_every == 0:
            timing = throughput.compute()

            logger.log(
                "step",
                step=global_step,
                epoch=epoch,
                batches_completed=batch_index + 1,
                losses=loss_values,
                loss=total_loss.detach().item(),
                lr=optimizer.param_groups[0]["lr"],
                images_per_second=timing["images_per_second"],
                peak_memory_mib=peak_memory_mb(reset=False),
                skipped_updates=skipped_updates,
            )

        if (
            checkpoint_fn is not None
            and global_step % checkpoint_every == 0
        ):
            checkpoint_fn(
                epoch=epoch,
                step=global_step,
                batches_completed=batch_index + 1,
            )

    logger.log(
        "epoch",
        step=global_step,
        epoch=epoch,
        skipped_updates=skipped_updates,
        peak_memory_mib=peak_memory_mb(reset=False),
    )

    return global_step