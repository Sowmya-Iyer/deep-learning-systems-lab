"""Experiment directories and metric logging."""

import json
import logging
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from dlsl.config import config_hash

logger = logging.getLogger(__name__)


def create_run_dir(cfg):
    """Create a unique directory for an experiment run."""
    runs_root = Path(cfg["output"]["runs_root"])
    experiment = cfg["experiment"]

    if (
        not isinstance(experiment, str)
        or not experiment.strip()
        or experiment in {".", ".."}
        or "/" in experiment
        or "\\" in experiment
    ):
        raise ValueError("experiment must be a nonempty folder name")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = runs_root / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Two runs may start during the same second.
    for index in range(10000):
        suffix = "" if index == 0 else f"-{index:03d}"
        run_dir = experiment_dir / f"{timestamp}{suffix}"

        try:
            run_dir.mkdir()
        except FileExistsError:
            continue

        return run_dir

    raise RuntimeError("Could not create a unique run directory")


class MetricLogger:
    """Append one JSON record per event."""

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self.path = run_dir / "metrics.jsonl"
        self._file = self.path.open(
            "a",
            encoding="utf-8",
        )

    def log(self, kind, step, **values):
        if kind not in {"step", "epoch", "eval"}:
            raise ValueError(f"Unknown metric kind: {kind!r}")

        if "time" in values:
            raise ValueError("'time' is reserved for the event timestamp")

        record = {
            "time": datetime.now().astimezone().isoformat(),
            "kind": kind,
            "step": step,
            **values,
        }

        # Reject NaN/Infinity instead of writing nonstandard JSON.
        line = json.dumps(record, allow_nan=False)

        self._file.write(line + "\n")
        self._file.flush()

        logger.info("%s step=%s %s", kind, step, values)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def _command_output(command, cwd=None):
    """Return command output, or None when unavailable."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.strip()


def write_environment(run_dir, cfg):
    """Record the environment and experiment settings."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Locate the repository containing this module.
    repo_root = next(
        (
            directory
            for directory in Path(__file__).resolve().parents
            if (directory / ".git").exists()
        ),
        None,
    )

    git_commit = None
    git_dirty = None

    if repo_root is not None:
        git_commit = _command_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
        )

        status = _command_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
        )

        # None means unknown; an empty string means a clean tree.
        git_dirty = None if status is None else bool(status)

    cuda_available = torch.cuda.is_available()
    gpus = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)

            gpus.append({
                "index": index,
                "name": properties.name,
                "total_memory_mib": properties.total_memory / (1024**2),
            })

    driver_output = _command_output([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])

    driver_versions = (
        sorted(set(driver_output.splitlines()))
        if driver_output
        else []
    )

    train_cfg = cfg.get("train", {})
    schedule_cfg = cfg.get("schedule", {})

    environment = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "os": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_versions": driver_versions,
        "gpus": gpus,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "precision": (
            "amp_float16"
            if cuda_available and train_cfg.get("amp", False)
            else "float32"
        ),
        "batch_size": train_cfg.get("batch_size"),
        "accum_steps": train_cfg.get("accum_steps", 1),
        "epochs": schedule_cfg.get("epochs"),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    path = run_dir / "environment.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2, allow_nan=False)

    return environment

class Throughput:
    """Measure images per second between reporting boundaries."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.reset()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self):
        self._synchronize()
        self._images = 0
        self._started = time.perf_counter()

    def update(self, image_count):
        """Count images without synchronizing or measuring time."""
        if image_count < 0:
            raise ValueError("image_count cannot be negative")

        self._images += image_count

    def compute(self):
        """Measure the completed interval and begin the next one."""
        self._synchronize()

        now = time.perf_counter()
        elapsed = now - self._started

        result = {
            "images": self._images,
            "seconds": elapsed,
            "images_per_second": (
                self._images / elapsed if elapsed > 0 else 0.0
            ),
        }

        self._images = 0
        self._started = now

        return result
def peak_memory_mb(reset=True):
    """Return peak allocated tensor memory in MiB on the current GPU."""
    if not torch.cuda.is_available():
        return 0.0

    peak = torch.cuda.max_memory_allocated() / (1024**2)

    if reset:
        torch.cuda.reset_peak_memory_stats()

    return peak