# Projects

The source of truth for what this repository must deliver. Each project applies one part of the learning path to forklift perception in a warehouse, and each is measured against a classical or simpler baseline. The [roadmap](README.md) sets the schedule; folder READMEs say where the work lives.

| # | Project | Weeks | Code | Report |
| - | ------- | ----- | ---- | ------ |
| 1 | Warehouse Object Recognition & ViT Scaling | 3-8, 19-20 | [training-pipeline](01-pytorch-fundamentals/training-pipeline/), [03-transformers](03-transformers/), [04-distributed](04-distributed/) | [report](reports/project-1-vit-scaling/report.md) |
| 2 | Label-Efficient Warehouse Perception | 9-12 | [05-self-supervised](05-self-supervised/) | [report](reports/project-2-label-efficient/report.md) |
| 3 | Learned Forklift Localization | 13-16, 18 | [06-multimodal/video-sensor](06-multimodal/video-sensor/) | [report](reports/project-3-video-sensor/report.md) |
| 4 | Aisle and Rack Label Reading | 13, 16 | [06-multimodal/vision-language](06-multimodal/vision-language/) | [report](reports/project-4-vision-language/report.md) |
| 5 | Warehouse 3D Perception | 17-18, 21-24 | [07-3d-vision](07-3d-vision/) | [report](reports/project-5-warehouse-3d/report.md) |

## Rules for every project

- **Data.** Use only public or synthetic data, or data with written permission from its owner. No employer footage, code, parameters or customer site data.
- **Baselines.** Classical baselines are rebuilt here from published methods.
- **Safety.** Learned perception here is advisory: earlier warnings, classification and slowdowns. It is not a safety function, and stopping stays with a safety-rated system.
- **Measurement.** Every number states its hardware, precision, batch size, training budget and seeds. Laptop and cloud results are never compared directly.

---

## Project 1: Warehouse Object Recognition & ViT Scaling

**Objective.** Implement, train, debug, profile and scale a modern deep learning model on warehouse objects.

**Research question.** How do architecture, training choices and hardware utilization affect recognition quality and training cost for warehouse objects?

**Build: foundation.** Create a reusable PyTorch pipeline with configs, data splits, train/eval, logging, checkpoints, resume and reproducible evaluation. Establish a CNN/ResNet baseline on crops of forklifts, pallet trucks, pallets and stillages from LOCO.

**Build: models.** Implement attention, multi-head attention, transformer blocks, patch embeddings and a small ViT. Compare scratch training and pretrained fine-tuning under a clear protocol.

**Build: systems.** Measure AMP, batch size, data loading and gradient accumulation. Add DDP in weeks 19-20; verify sampler/checkpoint behavior. Measure activation checkpointing; FSDP is conditional.

**Metrics.** Per-class and overall validation/test accuracy; training time; samples/second; peak GPU memory; scaling efficiency.

**Completion criteria.** A fresh user can reproduce a baseline, resume training and inspect fair comparisons. Explain what improved and its tradeoffs. The small ViT need not beat a pretrained model.

**Scope and interview defense.** Classification first; real detection uses pretrained detectors. LOCO has few forklift annotations, so report per-class results. Multi-GPU claims require measured runs on rented GPUs. Explain unstable training, bottlenecks, numerical behavior and distributed choices. One focused CUDA kernel is an optional extension.

---

## Project 2: Label-Efficient Warehouse Perception

**Objective.** Reduce the labels needed to bring up perception on a warehouse dataset.

**Research question.** Does self-supervised pretraining on warehouse images improve downstream performance at 10%, 25% and 100% of training labels?

**Build: baselines.** Freeze train/validation/test splits. Train supervised baselines at each label budget. Pretrain one manageable contrastive encoder on training images without labels.

**Build: evaluation.** Linear-probe the frozen encoder, then fine-tune it. Compare with an existing pretrained visual encoder. Keep downstream splits and evaluation rules consistent.

**Build: ablations.** Vary one or two factors such as augmentation strength or pretraining duration. Use at least 3 seeds for key comparisons when feasible; record any compute limitation.

**Metrics.** Accuracy or macro-F1 versus label budget; variation across seeds; pretraining cost; class-level errors; optional domain-shift evaluation.

**Completion criteria.** A reproducible report states whether pretraining helped, by how much and under which conditions. Keep validation/test images out of pretraining unless explicitly defining a transductive study.

**Scope and interview defense.** Complete one method deeply. MAE, DINO training and auto-labeling are extensions. Defend the hypothesis, controls, negative results and uncertainty.

---

## Project 3: Learned Forklift Localization

**Objective.** Establish whether learned measurements, a better state estimator and learned multimodal fusion each improve a classical localization stack.

**Research question.** When do learned measurements and a factor graph improve on ORB + EKF, and what happens when vision or odometry is degraded, missing or misaligned?

**Build: baseline.** Rebuild ORB matching against a reference map and EKF fusion with odometry. Measure pose error and lost-localization rate on the dataset chosen in the [data audit](reports/data-audit.md).

**Build: learned front end.** Add learned place recognition and feature matching, and feed them to the same EKF as measurements with fixed covariance.

**Build: factor graph.** Replace the EKF with a sliding-window factor graph with outlier rejection, using GTSAM. It must handle repeated aisles, false matches and late measurements better than the filter, or the report explains why not. Do not replace state estimation with an end-to-end pose regressor.

**Build: learned fusion.** Train a small model on cached visual features and an odometry window, with temporal positions and one cross-attention block. It predicts the reliability of each visual measurement, which sets its noise in the factor graph.

**Build: robustness.** Compare motion blur, low texture, wheel slip, dropped camera or odometry, and camera-to-odometry time offsets. Keep the route segments used for tuning and evaluation separate. If time runs short, keep the three most informative tests.

**Comparison.** All four versions use the same routes and tests:

| Version | Vision input | State estimator | Trust in vision |
| ------- | ------------ | --------------- | --------------- |
| Classical | ORB | EKF | Fixed |
| Learned front end | Learned matching | EKF | Fixed |
| Upgraded back end | Learned matching | Factor graph | Fixed, with outlier rejection |
| Full system | Learned matching | Factor graph | Predicted by cross-attention |

**Metrics.** Pose error; lost-localization rate; recovery time; behavior under each degradation; runtime per frame.

**Completion criteria.** Compare all four versions with the same protocol. Explain what each change contributed, and where it helps, fails or learns shortcuts. Save reproducible configs and a standalone report.

**Scope and interview defense.** No new data collection. If the audit finds no usable odometry, simulate it from ground-truth poses with documented noise, and say so in the report. Using the EKF as a fast filter beside the graph, and a particle filter for global localization, are extensions. Defend time alignment, aisle aliasing, noise models and what the system trusts.

---

## Project 4: Aisle and Rack Label Reading

**Objective.** Adapt a small pretrained vision-language model to read warehouse labels, and evaluate grounding and generalization.

**Research question.** Can parameter-efficient adaptation read aisle and rack labels better than an OCR baseline while preserving quality on a broader holdout?

**Build: baseline.** Output structured JSON such as `{aisle, bay, level}`. Fix prompts, preprocessing, decoding and evaluation. Run an OCR baseline and pretrained SmolVLM. Check model/data licenses, duplicates and split integrity.

**Build: adaptation.** Prepare image-text examples, synthetic where needed. Fine-tune with LoRA; verify trainable parameters, loss masking and tiny-batch overfit.

**Build: experiments.** Compare the OCR baseline, the pretrained model and one adaptation setting. Remove/shuffle images to test visual dependence. Evaluate a separate general holdout and audit hallucinations.

**Metrics.** Exact-match accuracy; general-holdout quality; hallucinated labels; trainable parameter count; peak memory; training time.

**Completion criteria.** Provide repeatable training/evaluation, adapter checkpoint or reproduction instructions, and an honest report of improvements and regressions with representative error examples.

**Scope and interview defense.** Compact project: week 13 baseline, week 16 adaptation and report. Training a VLM from scratch is outside scope. Feeding aisle readings into the Project 3 factor graph is an extension. Explain why a hallucinated label is a localization failure.

---

## Project 5: Warehouse 3D Perception

**Objective.** Connect classical geometry, neural scene representations and learned 3D perception on warehouse scenes.

**Research question.** Can a reconstructed warehouse scene supply useful training data, and how well can a small model recover depth and occupancy around a forklift?

**Build: geometry.** Implement camera models, then small-scale Structure from Motion and bundle adjustment on a warehouse sequence.

**Build: scene representation.** Train 3D Gaussian Splatting from those poses. Compare a NeRF only if time allows.

**Build: perception.** Train warehouse depth and occupancy prediction, with and without splat-rendered views as extra training data. Temporal tracking of forklifts and people is the first item dropped if time runs short.

**Build: deployment.** Export and quantize one model, then measure latency and accuracy.

**Metrics.** Reprojection and pose error; novel-view quality on held-out frames; depth and occupancy error; gain from rendered data; latency, model size and memory.

**Completion criteria.** One demo runs from warehouse images to a 3D scene and a perception output, with a report that states what worked, including any zero gain from rendered data.

**Scope and interview defense.** One sequence is enough, and it needs sufficient camera overlap. Explain reprojection error, gauge freedom, why splatting needs good poses, and what quantization cost. Near-miss prediction on synthetic scenes is an extension.
