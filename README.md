# deep-learning-systems-lab
Hands-on implementations and experiments in PyTorch, vision transformers, self-supervised learning, multimodal perception and distributed training, applied to warehouse perception.

This repository documents a structured 24-week progression through:

**PyTorch → Transformers → Self-Supervised Learning → Learned Localization → Vision-Language → Distributed Training → Interview Readiness**

Each study applies a modern training technique to a real warehouse problem, then measures whether it beats a simpler baseline. The goal is to understand how modern vision models are **implemented, trained, scaled, and evaluated honestly**.

---

## 🎯 Objectives

By the end of this lab, I aim to be able to:

* Write production-quality training pipelines in native PyTorch
* Implement Transformer and Vision Transformer components from first principles
* Design controlled experiments, with fair baselines, seeds and honest claims
* Reduce labeling needs through self-supervised pretraining
* Improve visual localization with learned measurements and cross-attention trust modeling
* Adapt a small vision-language model to read warehouse labels without hallucinating
* Train across multiple GPUs using DDP, and diagnose memory and data-loading bottlenecks

---

## 🗺️ Roadmap

18 hours a week, with 12 for training and projects and 6 for coding practice and career work. The full plan, including coding practice and applications, is tracked outside this repository.

| Weeks | Dates | Main work | Compute | Finished when |
| ----: | ----- | --------- | ------- | ------------- |
| 01–02 | Done | Tensors, autograd, modules and optimization | Laptop | Done |
| 03 | Nov 2–8 | Data audit; reproducible LOCO crop pipeline | Laptop | [Data audit](reports/data-audit.md) is complete; resume is verified |
| 04 | Nov 9–15 | ResNet baseline and failure analysis | Laptop | A fresh clone reproduces the baseline |
| 05–06 | Nov 16–29 | Attention, transformer blocks and small ViT | Laptop | Tests pass; controlled CNN comparison is done |
| 07–08 | Nov 30–Dec 13 | Pretrained adaptation; AMP and data-loading profiling | Laptop | First tables are in the Project 1 report |
| 09–10 | Dec 14–27 | Frozen splits, baselines and contrastive pretraining | Rented GPU | Baselines and encoder are saved |
| 11–12 | Dec 28–Jan 3 | Linear probes, one ablation and the Project 2 report | Rented GPU | Project 2 report is done |
| 13 | Jan 4–10 | Localization and label-reading baselines | Laptop | Baselines have measured results |
| 14 | Jan 11–17 | One learned visual measurement | Laptop first | Classical and learned are compared |
| 15 | Jan 18–24 | Localization robustness | Laptop | Simulated-corruption table is finished |
| 16 | Jan 25–31 | Label-reading LoRA adaptation | Laptop first | Tiny-batch overfit and comparison are done |
| 17 | Feb 1–7 | Cross-attention trust model and label evaluation | Laptop first | Trust ablation and real-image holdout are done |
| 18 | Feb 8–14 | Project 3 and 4 reports | Laptop | Both reports are done |
| 19–20 | Feb 15–28 | Two-GPU DDP and one memory study | Rented, 2 GPUs in one machine | Project 1 report is done |
| 21–24 | Mar 1–28 | ML coding, system design, project defense and applications | Laptop | Mock loop and readiness review are done |

Weeks 3–10 moved one week later, and weeks 11–12 share one week to get back on schedule. That leaves no spare time before January, so reduce runs or model size before cutting evaluation.

### Compute

Development and small experiments run on a laptop RTX 4070 (8 GB). Project 2 and DDP run on rented GPUs, at an estimated $100–200 in total. Check live prices before renting. Checkpoint and resume must work before week 9, because cheap instances can be interrupted. Datasets and runs stay outside OneDrive.

---

# 📊 Progress

| Weeks | Milestone | Status |
| ----: | --------- | :----: |
| 01–02 | Tensors, autograd, modules and optimization | ✅ |
| 03–04 | Data audit, pipeline and ResNet baseline | ⬜ |
| 05–08 | Small ViT and training measurements | ⬜ |
| 09–12 | Project 2 report | ⬜ |
| 13–18 | Project 3 and 4 reports | ⬜ |
| 19–20 | DDP scaling and Project 1 report | ⬜ |
| 21–24 | Interview readiness | ⬜ |

---

# 📈 Experiment Philosophy

The goal of each exercise is not simply to produce working code.

Where appropriate, experiments will measure:

* Training throughput
* GPU utilization
* GPU memory consumption
* Time to target quality
* Model accuracy, per class as well as overall
* Inference latency
* Data-loading performance

Failures, bottlenecks, unexpected results, and debugging observations are documented alongside successful experiments. A baseline that wins is a valid result.

---

# 🔬 Major Projects

The weekly exercises feed into four studies. [PROJECTS.md](PROJECTS.md) defines the objectives, metrics and completion criteria for each.

| Project | Research question | Code | Report |
| ------- | ----------------- | ---- | ------ |
| Warehouse Recognition & ViT Scaling | How do CNN and ViT training choices affect warehouse object classification, generalization and training cost? | `01`, `03`, `04` | [report](reports/project-1-vit-scaling/report.md) |
| Label-Efficient Warehouse Perception | Does warehouse-domain self-supervised pretraining improve downstream performance over scratch training and an external pretrained encoder at 10%, 25% and 100% of training labels? | `05` | [report](reports/project-2-label-efficient/report.md) |
| Robust Warehouse Localization with Learned Visual Measurements | Can a learned matching or relocalization component improve a fixed localization pipeline under visual degradation and sensor misalignment? | `06/video-sensor` | [report](reports/project-3-video-sensor/report.md) |
| Aisle and Rack Label Reading | Can LoRA adaptation read aisle and rack labels better than OCR and a pretrained VLM while preserving quality on unseen strings, layouts and real images? | `06/vision-language` | [report](reports/project-4-vision-language/report.md) |

One focused 3D study is an [optional extension](07-3d-vision/README.md) if time allows.

---

# 🗂️ Repository Structure

```text
deep-learning-systems-lab/
├── PROJECTS.md                  # what each project must deliver
├── 01-pytorch-fundamentals/     # tensors, autograd, production-style training
│   └── training-pipeline/       # shared pipeline, installed as the dlsl package
├── 02-cuda-kernels/             # CUDA notes and optional GPU exercises
├── 03-transformers/             # attention, transformer blocks, ViT
├── 04-distributed/              # DDP, memory analysis, optional FSDP
├── 05-self-supervised/          # label-efficient representation learning
├── 06-multimodal/
│   ├── video-sensor/            # learned localization and trust modeling
│   └── vision-language/         # aisle and rack label reading
├── 07-3d-vision/                # optional extension
└── reports/                     # data audit, tables, figures, project defenses
```

Code, configs and frozen splits are committed. Datasets, checkpoints and `runs/` are not; final numbers are copied into `reports/`.

---

# 📚 Primary References

### PyTorch

Official PyTorch documentation and tutorials, with emphasis on:

* training systems
* AMP
* profiling
* `torch.distributed`
* DDP

### GPU Programming

**Programming Massively Parallel Processors**
David B. Kirk & Wen-mei W. Hwu

Used alongside the optional CUDA exercises to develop intuition about GPU execution and memory hierarchy.

### Computer Vision

**Stanford CS231n — Deep Learning for Computer Vision**

Used selectively for modern vision architectures, self-supervised learning, video understanding and vision-language models.

### 3D Vision

**Stanford CS231A — Computer Vision: From 3D Reconstruction to Recognition**

Used for camera geometry, epipolar geometry, pose estimation, and reconstruction.

### Core Papers

* *Attention Is All You Need*
* *An Image Is Worth 16×16 Words*
* *A Simple Framework for Contrastive Learning of Visual Representations*
* *DINOv2: Learning Robust Visual Features without Supervision*
* *Learning Transferable Visual Models From Natural Language Supervision*
* *3D Gaussian Splatting for Real-Time Radiance Field Rendering*

---

# 🧠 Guiding Principle

The emphasis of this repository is **understanding by implementation**.

For each major concept, the preferred progression is:

```text
Learn
  ↓
Implement
  ↓
Test
  ↓
Profile
  ↓
Benchmark
  ↓
Break
  ↓
Debug
  ↓
Explain
```

The objective is not to collect implementations of existing models.

It is to understand **why the systems behave the way they do** — from tensor layouts and distributed communication to representation learning, sensor fusion, and honest evaluation.

---

## Status

🚧 **Active — 24-week study and engineering program**

Individual modules contain their own README files with implementation details, experiments, results, and lessons learned.
