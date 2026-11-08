import json

import pytest

from dlsl.runlog import create_run_dir, MetricLogger

from dlsl import runlog
from dlsl.runlog import (
    Throughput,
    peak_memory_mb,
    write_environment,
)


def test_environment_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runlog.torch.cuda,
        "is_available",
        lambda: False,
    )

    monkeypatch.setattr(
        runlog,
        "_command_output",
        lambda *args, **kwargs: None,
    )

    cfg = {
        "seed": 42,
        "train": {
            "batch_size": 2,
            "amp": False,
            "accum_steps": 1,
        },
        "schedule": {"epochs": 16},
    }

    write_environment(tmp_path, cfg)

    environment = json.loads(
        (tmp_path / "environment.json").read_text(encoding="utf-8")
    )

    required_keys = {
        "python",
        "os",
        "torch",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "driver_versions",
        "gpus",
        "git_commit",
        "git_dirty",
        "config_hash",
        "seed",
        "precision",
        "batch_size",
        "accum_steps",
        "epochs",
    }

    assert required_keys <= environment.keys()
    assert environment["cuda_available"] is False
    assert environment["gpus"] == []
    assert environment["precision"] == "float32"
    assert environment["batch_size"] == 2
    assert environment["epochs"] == 16


def test_throughput_counts_and_resets(monkeypatch):
    # Control the clock so the test does not need to sleep.
    times = iter([10.0, 12.0, 13.0])

    monkeypatch.setattr(
        runlog.time,
        "perf_counter",
        lambda: next(times),
    )

    throughput = Throughput("cpu")

    throughput.update(4)
    throughput.update(6)

    first = throughput.compute()

    assert first["images"] == 10
    assert first["seconds"] == 2.0
    assert first["images_per_second"] == 5.0

    throughput.update(3)
    second = throughput.compute()

    assert second["images"] == 3
    assert second["images_per_second"] == 3.0


def test_peak_memory_on_cpu(monkeypatch):
    monkeypatch.setattr(
        runlog.torch.cuda,
        "is_available",
        lambda: False,
    )

    assert peak_memory_mb() == 0.0

    
def test_run_directories_are_unique(tmp_path):
    cfg = {
        "experiment": "baseline",
        "output": {"runs_root": str(tmp_path)},
    }

    first = create_run_dir(cfg)
    second = create_run_dir(cfg)

    assert first.is_dir()
    assert second.is_dir()
    assert first != second
    assert first.parent == tmp_path / "baseline"
    assert second.parent == tmp_path / "baseline"


def test_metrics_are_valid_json_lines(tmp_path):
    with MetricLogger(tmp_path) as metrics:
        metrics.log("step", step=1, loss=1.5)
        metrics.log("eval", step=2, map_main=0.25)

        # Records must be readable before the logger closes.
        lines = metrics.path.read_text(
            encoding="utf-8"
        ).splitlines()

    records = [json.loads(line) for line in lines]

    assert len(records) == 2
    assert records[0]["kind"] == "step"
    assert records[0]["step"] == 1
    assert records[0]["loss"] == 1.5
    assert records[1]["map_main"] == 0.25
    assert all("time" in record for record in records)


def test_reopening_preserves_previous_records(tmp_path):
    with MetricLogger(tmp_path) as metrics:
        metrics.log("step", step=1, loss=1.5)

    with MetricLogger(tmp_path) as metrics:
        metrics.log("step", step=2, loss=1.0)

    lines = (tmp_path / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2


def test_nonfinite_metric_is_rejected(tmp_path):
    with MetricLogger(tmp_path) as metrics:
        with pytest.raises(ValueError):
            metrics.log("step", step=1, loss=float("nan"))

    assert (tmp_path / "metrics.jsonl").read_text(
        encoding="utf-8"
    ) == ""