import torch
from torch import nn

from dlsl.seed import seed_everything


def five_step_run(seed):
    seed_everything(seed, deterministic=True)

    # Fixed data isolates randomness in model initialization.
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X + 1

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []

    for _ in range(5):
        optimizer.zero_grad()

        prediction = model(X)
        loss = ((prediction - y) ** 2).mean()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test_same_seed_reproduces_losses():
    assert five_step_run(42) == five_step_run(42)


def test_different_seeds_change_losses():
    assert five_step_run(42) != five_step_run(123)