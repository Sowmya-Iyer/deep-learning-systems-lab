# deep-learning-systems-lab
Hands-on implementations and experiments in PyTorch, vision transformers, distributed training, self-supervised learning, multimodal perception and 3D vision, applied to forklift perception in warehouses.

This repository documents a structured 24-week progression through:

**PyTorch → Transformers → Self-Supervised Learning → Learned Localization → Vision-Language → 3D Vision → Distributed Training → Deployment**

Each stage replaces or extends a classical perception method, then measures whether the learned approach is actually better. The goal is to understand how modern vision models are **implemented, trained, scaled, optimized, and deployed in real-world perception systems**.

---

## 🎯 Objectives

By the end of this lab, I aim to be able to:

* Write production-quality training pipelines in native PyTorch
* Implement Transformer and Vision Transformer components from first principles
* Reduce labeling needs through self-supervised pretraining
* Improve visual localization with learned measurements, factor-graph estimation and cross-attention fusion
* Adapt a small vision-language model to read warehouse labels without hallucinating
* Implement multi-view geometry, Structure from Motion and bundle adjustment
* Reconstruct warehouse scenes with 3D Gaussian Splatting
* Train depth and occupancy models for scene awareness
* Train across multiple GPUs using DDP, and diagnose memory and data-loading bottlenecks
* Deploy a quantized model and measure its latency

---

## 🗺️ Roadmap

| Weeks | Phase | Main work | Compute | Finished when |
| ----: | ----- | --------- | ------- | ------------- |
| 01–02 | Foundations and data audit | Finish tensors and autograd; check sensors, licenses and ground truth for each dataset | Laptop | [Data audit](reports/data-audit.md) is complete |
| 03–04 | Training pipeline | Configs, checkpoints and resume; ResNet baseline on LOCO crops | Laptop | A fresh clone reproduces the baseline |
| 05–06 | Transformers | Attention, multi-head attention and transformer blocks, with tests | Laptop | Tests pass and a tiny batch overfits |
| 07–08 | Small ViT | Scratch vs pretrained; measure AMP, batch size and data loading | Laptop | First comparison tables are in the Project 1 report |
| 09–10 | Label-efficient learning | Freeze splits; baselines at each label budget; contrastive pretraining | Rented GPU | Baselines and encoder are saved |
| 11–12 | Label-efficient evaluation | Linear probe, fine-tuning and ablations with 3 seeds | Rented GPU | Project 2 report is done |
| 13 | Localization setup | Rebuild ORB + EKF; run the aisle-label baseline | Laptop | Classical baseline has measured pose error |
| 14 | Learned front end | Place recognition and matching with the EKF; run robustness tests | Laptop first | Learned and classical results are compared |
| 15 | Factor graph | Sliding-window factor graph with outlier rejection, using GTSAM | Laptop | Graph and EKF results are compared |
| 16 | Fusion and aisle labels | Cross-attention reliability model; compact SmolVLM adaptation and hallucination audit | Laptop first | Full system is measured; Project 4 report is done |
| 17 | Geometry | Camera models, Structure from Motion and bundle adjustment | Laptop | Reprojection and pose error are measured |
| 18 | Scene representation | 3D Gaussian Splatting from those poses | Laptop first | Novel views are scored; Project 3 report is done |
| 19–20 | Distributed training | DDP, sampler and checkpoint checks; scaling measurements | Rented, 2–4 GPUs in one machine | Project 1 report is done |
| 21–22 | Warehouse 3D perception | Depth and occupancy, with and without rendered views | Rented GPU | Gain from rendered data is measured |
| 23 | Deployment | Export and quantize one model; measure latency and accuracy | Laptop or edge device | Deployment table is finished |
| 24 | Integration | Demo, final report and defense notes | Laptop | Project 5 report is done and the demo runs |

Assumes 15–20 hours a week. Week 16 is the tightest. If time runs short, drop temporal tracking first, then the NeRF comparison.

### Compute

Development and small experiments run on a laptop RTX 4070 (8 GB). Heavy training runs on rented GPUs, at an estimated $150–400 for the whole roadmap. Check live prices before renting. Checkpoint and resume must work before week 9, because cheap instances can be interrupted. Datasets and runs stay outside OneDrive.

---

# 📊 Progress

| Weeks | Milestone | Status |
| ----: | --------- | :----: |
| 01–02 | Data audit | ⬜ |
| 03–04 | Pipeline and ResNet baseline | ⬜ |
| 05–06 | Attention and transformer blocks | ⬜ |
| 07–08 | Small ViT and training measurements | ⬜ |
| 09–12 | Project 2 report | ⬜ |
| 13–16 | Localization comparison | ⬜ |
| 16 | Project 4 report | ⬜ |
| 17–18 | SfM, Gaussian Splatting, Project 3 report | ⬜ |
| 19–20 | DDP scaling, Project 1 report | ⬜ |
| 21–22 | Depth and occupancy | ⬜ |
| 23 | Deployment benchmark | ⬜ |
| 24 | Project 5 report and demo | ⬜ |

---

# 📈 Experiment Philosophy

The goal of each exercise is not simply to produce working code.

Where appropriate, experiments will measure:

* Training throughput
* GPU utilization
* GPU memory consumption
* Scaling efficiency
* Model accuracy, per class as well as overall
* Inference latency
* Model size
* Data-loading performance

Failures, bottlenecks, unexpected results, and debugging observations are documented alongside successful experiments. A classical baseline that wins is a valid result.

---

# 🔬 Major Projects

The weekly exercises feed into five projects. [PROJECTS.md](PROJECTS.md) defines the objectives, metrics and completion criteria for each.

| Project | Research question | Code | Report |
| ------- | ----------------- | ---- | ------ |
| Warehouse Object Recognition & ViT Scaling | How do architecture, training choices and hardware utilization affect recognition quality and training cost for warehouse objects? | `01`, `03`, `04` | [report](reports/project-1-vit-scaling/report.md) |
| Label-Efficient Warehouse Perception | Does self-supervised pretraining on warehouse images improve downstream performance at 10%, 25% and 100% of training labels? | `05` | [report](reports/project-2-label-efficient/report.md) |
| Learned Forklift Localization | When do learned measurements and a factor graph improve on ORB + EKF, and what happens when vision or odometry is degraded, missing or misaligned? | `06/video-sensor` | [report](reports/project-3-video-sensor/report.md) |
| Aisle and Rack Label Reading | Can parameter-efficient adaptation read aisle and rack labels better than an OCR baseline while preserving quality on a broader holdout? | `06/vision-language` | [report](reports/project-4-vision-language/report.md) |
| Warehouse 3D Perception | Can a reconstructed warehouse scene supply useful training data, and how well can a small model recover depth and occupancy around a forklift? | `07` | [report](reports/project-5-warehouse-3d/report.md) |

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
│   ├── video-sensor/            # learned localization, factor graph, multimodal fusion
│   └── vision-language/         # aisle and rack label reading
├── 07-3d-vision/                # SfM, Gaussian Splatting, depth, occupancy, deployment
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
* *Masked Autoencoders Are Scalable Vision Learners*
* *Emerging Properties in Self-Supervised Vision Transformers*
* *DINOv2: Learning Robust Visual Features without Supervision*
* *Learning Transferable Visual Models From Natural Language Supervision*
* *NeRF: Representing Scenes as Neural Radiance Fields*
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

It is to understand **why the systems behave the way they do** — from tensor layouts and distributed communication to representation learning, sensor fusion, multi-view geometry, and real-time model deployment.

---

## Status

🚧 **Active — 24-week study and engineering program**

Individual modules contain their own README files with implementation details, experiments, results, and lessons learned.
