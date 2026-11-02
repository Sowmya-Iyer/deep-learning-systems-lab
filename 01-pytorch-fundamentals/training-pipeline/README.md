# training-pipeline

The reusable PyTorch pipeline for [Project 1](../../PROJECTS.md), shared by every later project.

The numbered repository folders are not valid Python module names, so the shared code is the installable package `dlsl`. Folders 03 to 06 import it instead of copying it.

## Install

With `.torch-venv` active, from `01-pytorch-fundamentals/`:

```
pip install -e training-pipeline
```

Then `import dlsl` works from anywhere in the repository. The package declares no dependencies on purpose, so installing it never replaces the CUDA build of PyTorch pinned in `../requirements.txt`.

## Layout

```
training-pipeline/
├── pyproject.toml
├── configs/          # one file per experiment, tracked in Git
└── src/dlsl/         # shared code
```

Planned modules in `src/dlsl/`:

| Module | Responsibility |
| ------ | -------------- |
| `config.py` | Load an experiment config; save the resolved copy into the run directory |
| `data.py` | Build COCO-format full-image datasets, crop datasets, and frozen train/validation/test splits |
| `metrics.py` | COCO mAP, per-class AP, recall and false positives per image |
| `engine.py` | Train and evaluation loops, AMP, gradient accumulation |
| `checkpoint.py` | Save and resume model, optimizer, scheduler, scaler, RNG and epoch |
| `logging.py` | Per-step and per-epoch metrics, images/second, peak GPU memory |
| `seed.py` | Seeding and deterministic settings |

## Conventions

- **Runs.** Each run writes to `runs/<experiment>/<timestamp>/`: resolved config, metrics, checkpoints, and the hardware, precision, batch size and training budget. `runs/` is not committed; final numbers are copied into [`reports/`](../../reports/).
- **Data.** Keep datasets and runs outside OneDrive. Read the location from a config value or a `DATA_ROOT` environment variable, not a hard-coded path, so the same config runs on the laptop and on rented GPUs.
- **Rented GPUs.** Instances can be interrupted and their disks disappear. Resume must work before the first rented run, and checkpoints must be copied off the instance as training runs.

## Done when

- [ ] A pretrained Faster R-CNN baseline trains on full LOCO images from a single config
- [ ] Interrupted training resumes and continues the same run
- [ ] Evaluation of a saved checkpoint reproduces the logged metrics
- [ ] A fresh clone can do all of the above from this README
