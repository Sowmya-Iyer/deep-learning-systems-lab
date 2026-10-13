# deep-learning-systems-lab
Hands-on implementations and experiments in GPU programming, PyTorch, vision transformers, distributed training, self-supervised learning, multimodal perception, and 3D vision.

This repository documents a structured 24-week progression through:

**CUDA → PyTorch → Transformers → Distributed Training → Self-Supervised Learning → Video & Multimodal Models → 3D Vision → Efficient Deployment**

Rather than treating these topics as isolated tutorials, the goal is to understand how modern vision and foundation models are **implemented, trained, scaled, optimized, and ultimately deployed in real-world perception systems**.

---

## 🎯 Objectives

By the end of this lab, I aim to be able to:

* Write production-quality training pipelines in native PyTorch
* Understand how deep-learning workloads execute on GPUs
* Implement Transformer and Vision Transformer components from first principles
* Train models across multiple GPUs using DDP and FSDP
* Diagnose GPU memory, utilization, data-loading, and communication bottlenecks
* Understand and implement modern self-supervised vision techniques
* Build temporal and multimodal perception models
* Develop practical understanding of multi-view geometry and 3D reconstruction
* Optimize trained models through quantization, distillation, and deployment tooling
* Reason about the complete lifecycle of a large-scale robotics perception model

---

## 🗺️ Roadmap

| Phase | Weeks | Focus                                |
| ----- | ----: | ------------------------------------ |
| I     | 01–04 | PyTorch & GPU Foundations            |
| II    | 05–08 | Transformers & Vision Transformers   |
| III   | 09–12 | Distributed & Large-Scale Training   |
| IV    | 13–16 | Self-Supervised & Foundation Vision  |
| V     | 17–20 | Video, Multimodal & Efficient Models |
| VI    | 21–24 | 3D Vision & ML Systems               |

---

## 01 — PyTorch & GPU Foundations

**Weeks 1–4**

Build the foundations of GPU-aware deep-learning engineering.

### Topics

* PyTorch tensors and tensor memory layout
* CPU ↔ GPU execution
* Autograd and computational graphs
* `nn.Module` and model construction
* Dataset and DataLoader design
* Training and evaluation loops
* Optimizers and learning-rate schedules
* Checkpointing and reproducibility
* CUDA threads, blocks, grids, and warps
* GPU memory hierarchy
* Coalesced memory access
* Shared-memory tiling
* Synchronization and atomics

### Exercises

* CPU vs CUDA vs PyTorch vector addition
* Manual linear regression with autograd
* CUDA image-processing kernel
* Naive CUDA matrix multiplication
* Shared-memory tiled matrix multiplication
* Production-style CIFAR training pipeline

---

## 02 — Transformers & Vision Transformers

**Weeks 5–8**

Move from tensor operations to modern neural architectures.

### Topics

* Query, Key, and Value representations
* Scaled dot-product attention
* Self-attention and cross-attention
* Multi-head attention
* Attention masks
* Layer normalization
* Residual connections
* Transformer encoder blocks
* Patch embeddings
* Positional embeddings
* Vision Transformers
* Mixed-precision training
* Gradient accumulation and clipping
* PyTorch profiling
* Training throughput optimization

### Exercises

* Single-head attention from scratch
* Multi-head attention from scratch
* Transformer encoder
* Vision Transformer implementation
* CIFAR-100 ViT training
* ViT profiling and optimization

---

## 03 — Distributed & Large-Scale Training

**Weeks 9–12**

Understand how training changes when one GPU is no longer enough.

### Topics

* Processes, ranks, and world size
* NCCL
* Distributed collectives
* `AllReduce`
* `AllGather`
* `ReduceScatter`
* DistributedDataParallel
* Distributed sampling
* Scaling efficiency
* Model and optimizer memory
* Fully Sharded Data Parallel
* Parameter and optimizer-state sharding
* Activation checkpointing
* BF16 mixed precision
* Gradient accumulation
* Tensor parallelism
* Pipeline parallelism
* Compute/communication tradeoffs

### Exercises

* Distributed collective communication demo
* Single-GPU → multi-GPU ViT
* DDP throughput benchmark
* DDP vs FSDP memory comparison
* Large-model OOM investigation
* Memory reduction using FSDP + BF16 + activation checkpointing

---

## 04 — Self-Supervised & Foundation Vision

**Weeks 13–16**

Explore how large visual representations can be learned from unlabeled data.

### Topics

* Representation learning
* Contrastive learning
* Self-supervised pretraining
* Linear probing
* Masked image modeling
* MAE
* Teacher/student learning
* EMA teachers
* DINO/DINOv2 concepts
* CLIP-style vision-language learning
* Pseudo-labeling
* Confidence filtering
* Automatic labeling pipelines
* Hard-example mining

### Exercises

* Contrastive visual encoder
* Embedding visualization
* MAE pretraining experiment
* Pretrained vs random initialization comparison
* Teacher/student SSL experiment
* Linear probing
* Automated video labeling pipeline

---

## 05 — Video, Multimodal & Efficient Models

**Weeks 17–20**

Extend perception from individual images to temporal and multimodal environments.

### Topics

* Video tensors and frame sampling
* Temporal positional embeddings
* Spatial vs temporal attention
* Video Transformers
* Early and late multimodal fusion
* Cross-attention
* Modality-specific encoders
* Camera + state/IMU representations
* World-model concepts
* Action conditioning
* Future representation prediction
* Vision-language-action models
* Knowledge distillation
* Quantization
* PTQ and QAT
* Structured and unstructured pruning

### Exercises

* Video classification with temporal attention
* Camera + IMU/state fusion Transformer
* Future-state prediction
* Action-conditioned latent prediction
* Teacher/student distillation
* FP16/INT8 deployment benchmark
* ONNX/TensorRT export

---

## 06 — 3D Vision & ML Systems

**Weeks 21–24**

Connect learned perception with geometry and complete-system reasoning.

### Topics

* Camera coordinate systems
* Intrinsic and extrinsic calibration
* Homogeneous coordinates
* Camera projection
* Fundamental and essential matrices
* Epipolar geometry
* PnP
* Triangulation
* Stereo geometry
* Structure from Motion
* Bundle adjustment
* NeRF concepts
* 3D Gaussian Splatting
* Large-scale ML data engines
* Model lifecycle design
* Training/deployment feedback loops

### Exercises

* Feature matching and epipolar-line visualization
* Camera pose estimation
* Multi-view triangulation
* 3D point-cloud reconstruction
* Small neural/3D scene representation experiment
* End-to-end foundation-model system design

---

# 📊 Progress

| Week | Focus                    | Implementation / Experiment             | Status |
| ---: | ------------------------ | --------------------------------------- | :----: |
|   01 | PyTorch tensors + CUDA   | Vector-add benchmark                    |    ⬜   |
|   02 | Autograd + GPU execution | Manual regression + CUDA image kernel   |    ⬜   |
|   03 | GPU memory               | Naive vs tiled GEMM                     |    ⬜   |
|   04 | Training systems         | CIFAR-100 training framework            |    ⬜   |
|   05 | Attention                | Scaled dot-product attention            |    ⬜   |
|   06 | Transformers             | Multi-head attention + encoder          |    ⬜   |
|   07 | Vision Transformers      | ViT from scratch                        |    ⬜   |
|   08 | Training optimization    | AMP + profiling                         |    ⬜   |
|   09 | Distributed fundamentals | Collective communication                |    ⬜   |
|   10 | DDP                      | Distributed ViT                         |    ⬜   |
|   11 | FSDP                     | DDP vs FSDP benchmark                   |    ⬜   |
|   12 | Large-model training     | Memory optimization                     |    ⬜   |
|   13 | SSL                      | Contrastive encoder                     |    ⬜   |
|   14 | MAE                      | Masked pretraining                      |    ⬜   |
|   15 | DINO                     | Teacher/student learning                |    ⬜   |
|   16 | Data engines             | Automatic labeling pipeline             |    ⬜   |
|   17 | Video                    | Temporal Transformer                    |    ⬜   |
|   18 | Multimodal               | Vision + state/IMU fusion               |    ⬜   |
|   19 | World models             | Future representation prediction        |    ⬜   |
|   20 | Model efficiency         | Distillation + quantization             |    ⬜   |
|   21 | Geometry                 | Epipolar geometry                       |    ⬜   |
|   22 | 3D                       | Multi-view reconstruction               |    ⬜   |
|   23 | ML systems               | Large-scale training/data-engine design |    ⬜   |
|   24 | Integration              | Final system + technical retrospective  |    ⬜   |

---

# 📈 Experiment Philosophy

The goal of each exercise is not simply to produce working code.

Where appropriate, experiments will measure:

* Training throughput
* GPU utilization
* GPU memory consumption
* Scaling efficiency
* Model accuracy
* Inference latency
* Model size
* Data-loading performance

For example:

| GPUs | Global Batch | Throughput | Peak Memory/GPU | Scaling Efficiency |
| ---: | -----------: | ---------: | --------------: | -----------------: |
|    1 |            — |          — |               — |               100% |
|    2 |            — |          — |               — |                  — |
|    4 |            — |          — |               — |                  — |

Failures, bottlenecks, unexpected results, and debugging observations are documented alongside successful experiments.

---

# 🔬 Major Projects

The weekly exercises progressively feed into four larger projects.

### Distributed Vision Transformer

Build a Vision Transformer from first principles and evolve its training stack from single-GPU execution to:

**AMP → DDP → FSDP → activation checkpointing → distributed benchmarks**

### Robotics Vision Foundation Model

Develop a vision pretraining/data-engine pipeline around:

**unlabeled visual data → automatic labeling → self-supervised pretraining → downstream dense prediction**

### Multimodal Robot Perception

Explore temporal and multimodal representations using:

**video + motion/state signals → modality encoders → multimodal fusion → future prediction**

### Efficient Robot Perception

Take a trained vision model through:

**teacher → student → distillation → quantization → ONNX → TensorRT → latency/accuracy evaluation**

---

# 📚 Primary References

### GPU Programming

**Programming Massively Parallel Processors**
David B. Kirk & Wen-mei W. Hwu

Used alongside the CUDA exercises to develop intuition about GPU execution, memory hierarchy, tiling, synchronization, and parallel algorithms.

### PyTorch

Official PyTorch documentation and tutorials, with emphasis on:

* training systems
* AMP
* profiling
* `torch.distributed`
* DDP
* FSDP

### Computer Vision

**Stanford CS231n — Deep Learning for Computer Vision**

Used selectively for modern vision architectures, self-supervised learning, video understanding, vision-language models, and related topics.

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

It is to understand **why the systems behave the way they do** — from CUDA memory access and tensor layouts to distributed communication, representation learning, multimodal fusion, and real-time model deployment.

---

## Status

🚧 **Active — 24-week study and engineering program**

Individual modules contain their own README files with implementation details, experiments, results, and lessons learned.
>>>>>>> origin/main
