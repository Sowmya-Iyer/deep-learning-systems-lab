# Projects

The source of truth for what this repository must deliver. Four studies apply the learning path to warehouse perception, and each is measured against a simpler baseline. The [roadmap](README.md) sets the schedule; folder READMEs say where the work lives.

| # | Project | Weeks | Code | Report |
| - | ------- | ----- | ---- | ------ |
| 1 | Warehouse Recognition & ViT Scaling | 3-8, 19-20 | [training-pipeline](01-pytorch-fundamentals/training-pipeline/), [03-transformers](03-transformers/), [04-distributed](04-distributed/) | [report](reports/project-1-vit-scaling/report.md) |
| 2 | Label-Efficient Warehouse Perception | 9-12 | [05-self-supervised](05-self-supervised/) | [report](reports/project-2-label-efficient/report.md) |
| 3 | Robust Warehouse Localization with Learned Visual Measurements | 13-15, 17-18 | [06-multimodal/video-sensor](06-multimodal/video-sensor/) | [report](reports/project-3-video-sensor/report.md) |
| 4 | Aisle and Rack Label Reading | 13, 16-18 | [06-multimodal/vision-language](06-multimodal/vision-language/) | [report](reports/project-4-vision-language/report.md) |
| — | Optional 3D extension | When time allows | [07-3d-vision](07-3d-vision/) | Folder README |

## Rules for every project

- **Data.** Use only public or synthetic data, or data with written permission from its owner. No employer footage, code, parameters or customer site data.
- **Claims.** Name what was actually tested. Crop classification is not detection, injected faults are simulated, and localization on a mobile robot is not forklift localization.
- **Safety.** Learned perception here is advisory: earlier warnings, classification and slowdowns. It is not a safety function, and stopping stays with a safety-rated system.
- **Measurement.** Every number states its hardware, precision, batch size, training budget and seeds. Laptop and cloud results are never compared directly.

---

## Project 1: Warehouse Recognition & ViT Scaling

**Objective.** Implement, train, debug, profile and scale a modern deep learning model on warehouse objects.

**Research question.** How do CNN and ViT training choices affect warehouse object classification, generalization and training cost?

**Build: foundation.** Create a reusable PyTorch pipeline with configs, data splits, train/eval, logging, checkpoints, resume and reproducible evaluation. Establish a CNN/ResNet baseline on LOCO object crops. Split source images or sequences before generating crops, and audit class counts before choosing classes.

**Build: models.** Implement attention, multi-head attention, transformer blocks, patch embeddings and a small ViT. Compare pretrained CNN and ViT models separately, as a transfer-learning study rather than a pure architecture comparison.

**Build: systems.** Measure AMP, batch size, data loading and gradient accumulation. Add DDP in weeks 19-20 with two GPUs; verify sampler/checkpoint behavior. Four GPUs are conditional on a useful two-GPU result. Measure activation checkpointing; FSDP is conditional.

**Metrics.** Macro-F1; per-class recall; confusion matrices; accuracy; training time; samples/second; peak GPU memory; time-to-target quality; scaling efficiency; cost.

**Completion criteria.** A fresh user can reproduce a baseline, resume training and inspect fair comparisons. Explain what improved and its tradeoffs. The small ViT need not beat a pretrained model.

**Scope and interview defense.** This is object-crop classification, not full-scene detection. Multi-GPU claims require measured runs on rented GPUs. Explain unstable training, bottlenecks, numerical behavior and distributed choices. One focused CUDA kernel is an optional extension.

---

## Project 2: Label-Efficient Warehouse Perception

**Objective.** Design and evaluate representation-learning experiments when warehouse labels are limited.

**Research question.** Does warehouse-domain self-supervised pretraining improve downstream performance over scratch training and an external pretrained encoder at 10%, 25% and 100% of training labels?

**Build: baselines.** Freeze train/validation/test splits and nested, class-stratified label subsets. Train from scratch and fine-tune an externally pretrained encoder at each label budget. Pretrain one manageable contrastive encoder on training images without labels.

**Build: evaluation.** Linear-probe the frozen encoder, then fine-tune it. Keep downstream splits and evaluation rules consistent across all three approaches.

**Build: ablations.** Vary one or two factors such as augmentation strength or pretraining duration. Use at least 3 seeds for key comparisons when feasible; record any compute limitation.

**Metrics.** Accuracy or macro-F1 versus label budget; variation across seeds; pretraining cost; class-level errors; optional domain-shift evaluation.

**Completion criteria.** A reproducible report states whether pretraining helped, by how much and under which conditions. Keep validation/test images out of pretraining unless explicitly defining a transductive study.

**Scope and interview defense.** Complete one method deeply. MAE, DINO training and auto-labeling are extensions. Label efficiency within a domain and new-site adaptation are separate claims; make the second only with a held-out site. Defend the hypothesis, controls, negative results and uncertainty.

---

## Project 3: Robust Warehouse Localization with Learned Visual Measurements

**Objective.** Establish whether a learned visual measurement, and learned trust in that measurement, improve a fixed localization pipeline.

**Research question.** Can a learned matching or relocalization component improve a fixed localization pipeline under visual degradation and sensor misalignment?

**Build: baseline.** Build a reproducible classical baseline from existing components, with a fixed EKF. Use the dataset chosen in the [data audit](reports/data-audit.md), and establish reference-pose quality first.

**Build: learned measurement.** Replace one component: learned feature matching or place recognition. Keep the estimator fixed, so any gain comes from the measurement.

**Build: robustness.** Evaluate blur, visual dropout, time offsets and odometry corruption. Keep tuning and evaluation route segments separate.

**Build: learned trust.** Cache visual features and train a small model with temporal positions and one cross-attention block. It uses a visual and odometry window to set how much the EKF trusts each visual measurement. Drop this step if the baseline is not working by week 16.

**Comparison.** All versions use the same routes and tests:

| Version | Vision input | State estimator | Trust in vision |
| ------- | ------------ | --------------- | --------------- |
| Classical | Classical matching | EKF | Fixed |
| Learned measurement | Learned component | EKF | Fixed |
| Learned trust | Learned component | EKF | Predicted by cross-attention |

**Metrics.** Pose error against stated reference poses; lost-localization rate; recovery time; behavior under each corruption; runtime per frame.

**Completion criteria.** Compare all versions with the same protocol. Explain where each helps, fails or learns shortcuts. Save reproducible configs and a standalone report.

**Scope and interview defense.** No new data collection. Label injected wheel slip, delay and dropout as simulated, and describe reference poses by how they were produced. A factor graph, a particle filter and fusing aisle labels are extensions. Defend time alignment, reference quality, noise models and what the system trusts.

---

## Project 4: Aisle and Rack Label Reading

**Objective.** Adapt a small pretrained vision-language model to read warehouse labels, and evaluate grounding and generalization.

**Research question.** Can LoRA adaptation read aisle and rack labels better than OCR and a pretrained VLM while preserving quality on unseen strings, layouts and real images?

**Build: baseline.** Output structured `{aisle, bay, level}` fields. Fix prompts, preprocessing, decoding and evaluation. Run OCR with deterministic parsing and pretrained SmolVLM. Check model/data licenses, duplicates and split integrity.

**Build: adaptation.** Prepare image-text examples, synthetic where needed. Fine-tune with LoRA; verify trainable parameters, loss masking and tiny-batch overfit.

**Build: experiments.** Compare OCR, the pretrained VLM and LoRA. Test missing or unreadable labels and appropriate abstention, plus unseen strings, fonts, layouts and capture conditions. Remove/shuffle images to test visual dependence. Keep a small, fixed, separately sourced real-image holdout.

**Metrics.** Field-level and complete-record exact match; character error rate; schema validity, scored separately from content; abstention quality; hallucinated labels; trainable parameter count; peak memory; training time.

**Completion criteria.** Provide repeatable training/evaluation, adapter checkpoint or reproduction instructions, and an honest report of improvements and regressions with representative error examples.

**Scope and interview defense.** Small study: week 13 baseline, week 16 adaptation, week 17 evaluation, week 18 report. Training a VLM from scratch is outside scope. Synthetic-only evaluation shows synthetic-task performance only. Explain why a hallucinated label is a localization failure.

---

## Optional extension: one focused 3D study

Only with spare project time. Weeks 21-24 stay reserved for interview preparation.

**Question.** How does pose quality affect novel-view quality on one warehouse sequence?

**Build.** Run Structure from Motion and small-scale bundle adjustment, train 3D Gaussian Splatting, and compare pose sources on held-out views.

**Limit.** Good novel views do not prove accurate metric geometry or useful data augmentation, so make neither claim.
