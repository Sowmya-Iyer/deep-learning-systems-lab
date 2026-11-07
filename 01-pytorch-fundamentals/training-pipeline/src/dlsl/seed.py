import os
import random

import numpy as np
import torch

# A seed sets the starting state of a random-number generator. It reproduces a sequence of random values; it doesn’t make every value identical.
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed random generators and configure deterministic execution."""

    # Set before CUDA work begins.
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )

    torch.backends.cudnn.benchmark = False

# DataLoader workers run in separate processes. Give their NumPy and Python generators seeds derived from the worker seed PyTorch assigns

def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    """Create an independent CPU random generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

# Example Usage:
# loader = DataLoader(
#     dataset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=4,
#     worker_init_fn=seed_worker,
#     generator=make_generator(42),
# )

