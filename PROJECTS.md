# Projects

The source of truth for what this repository must deliver. Four studies apply the learning path to warehouse and driving perception, and each is measured against a simpler baseline. The [roadmap](README.md) sets the schedule; folder READMEs say where the work lives.

| # | Project | Weeks | Code | Report |
| - | ------- | ----- | ---- | ------ |
| 1 | Warehouse Detection & ViT Scaling | 3-8, 19-20 | [training-pipeline](01-pytorch-fundamentals/training-pipeline/), [03-transformers](03-transformers/), [04-distributed](04-distributed/) | [report](reports/project-1-vit-scaling/report.md) |
| 2 | Label-Efficient Free-Space Segmentation | 9-12 | [05-self-supervised](05-self-supervised/) | [report](reports/project-2-label-efficient/report.md) |
| 3 | Robust Driving Localization with Learned Visual Measurements | 13-15, 17-18 | [06-multimodal/video-sensor](06-multimodal/video-sensor/) | [report](reports/project-3-video-sensor/report.md) |
| 4 | Aisle and Rack Label Reading | 13, 16-18 | [06-multimodal/vision-language](06-multimodal/vision-language/) | [report](reports/project-4-vision-language/report.md) |
| — | Optional 3D extension | When time allows | [07-3d-vision](07-3d-vision/) | Folder README |

## Rules for every project

- **Data.** Use only public or synthetic data, or data with written permission from its owner. No employer footage, code, parameters or customer site data.
- **Claims.** Name what was actually tested. Crop classification is not detection, injected faults are simulated, and localization on a road car is not forklift localization.
- **Safety.** Learned perception here is advisory: earlier warnings, classification and slowdowns. It is not a safety function, and stopping stays with a safety-rated system.
- **Measurement.** Every number states its hardware, precision, batch size, training budget and seeds. Laptop and cloud results are never compared directly.

---

## Project 1: Warehouse Detection & ViT Scaling

**Objective.** Implement, train, debug, profile and scale modern deep learning models that find warehouse objects in full images.

**Research question.** How do CNN and transformer training choices affect warehouse object detection, generalization and training cost?

**Build: foundation.** Create a reusable PyTorch pipeline with configs, data splits, train/eval, logging, checkpoints, resume and reproducible evaluation. Establish a pretrained Faster R-CNN baseline on full LOCO images, using the frozen subset split in the [data audit](reports/data-audit.md). Check for unlabeled objects before trusting false-positive counts.

**Build: models.** Implement attention, multi-head attention, transformer blocks, patch embeddings and a small ViT, and test the ViT on object crops. Then compare Faster R-CNN with a pretrained DETR-family detector, as a transfer-learning study rather than a pure architecture comparison. Drop the DETR comparison before cutting evaluation if the schedule slips.

**Build: systems.** Measure AMP, batch size, data loading and gradient accumulation. Add DDP to the detector in weeks 19-20 with two GPUs; verify sampler/checkpoint behavior. Four GPUs are conditional on a useful two-GPU result. Measure activation checkpointing; FSDP is conditional.

**Metrics.** mAP@50; mAP@50:95; per-class AP; recall; small-object AP; false positives per image; inference latency; training time; images/second; peak GPU memory; time-to-target quality; scaling efficiency; cost.

**Completion criteria.** A fresh user can reproduce a baseline, resume training and inspect fair comparisons. Explain what improved and its tradeoffs. The small ViT need not beat a pretrained model.

**Scope and interview defense.** This is detection on five LOCO classes. Forklift is reported with a warning because it has few boxes and only 5 in validation, so it cannot be tuned on. Published LOCO scores use a different split, so they are a reference, not a direct comparison. Multi-GPU claims require measured runs on rented GPUs. Explain unstable training, bottlenecks, numerical behavior and distributed choices. One focused CUDA kernel is an optional extension.

---

## Project 2: Label-Efficient Free-Space Segmentation

**Objective.** Design and evaluate representation-learning experiments for free-space segmentation, where pixel labels are slow and expensive to make. Free space answers where a vehicle can safely go, which extends obstacle detection toward scene understanding.

**Research question.** Does driving-domain self-supervised pretraining improve free-space segmentation over scratch training and an external pretrained encoder at 10%, 25% and 100% of training labels?

**Build: baselines.** Use BDD100K drivable-area labels if the [data audit](reports/data-audit.md) confirms its terms. Freeze train, validation and test splits, and nested label subsets balanced across weather and time of day. Train from scratch and fine-tune an externally pretrained encoder at each label budget, with the same light segmentation head. Pretrain one manageable contrastive encoder on training images without labels. Use a reduced image resolution, so that 27 runs across approaches, budgets and seeds stay affordable.

**Build: evaluation.** Train only the segmentation head on the frozen encoder, then fine-tune the whole model. Keep downstream splits and evaluation rules consistent across all three approaches.

**Build: ablations.** Vary one or two factors such as augmentation strength or pretraining duration. Use at least 3 seeds for key comparisons when feasible; record any compute limitation.

**Metrics.** Free-space IoU versus label budget; boundary accuracy; free space wrongly predicted inside labeled vehicles and pedestrians; variation across seeds; pretraining cost; results by weather and time of day.

**Completion criteria.** A reproducible report states whether pretraining helped, by how much and under which conditions. Keep validation/test images out of pretraining unless explicitly defining a transductive study.

**Scope and interview defense.** Complete one method deeply. Free-space masks here are learned from labels, not checked against a vehicle's path, so this is not a driving safety function. If the 25% budget or a seed must go to save time, cut it before cutting evaluation. Fine-tuning the Project 1 detector from this encoder is an extension. MAE, DINO training and auto-labeling are extensions. Label efficiency and adaptation to new conditions are separate claims; make the second only with held-out conditions. Defend the hypothesis, controls, negative results and uncertainty.

---

## Project 3: Robust Driving Localization with Learned Visual Measurements

**Objective.** Establish whether a learned visual measurement, and learned trust in that measurement, improve a fixed localization pipeline on public driving data. This applies the warehouse approach of matching against a reference map, fused with odometry in an EKF, to data with stronger ground truth.

**Research question.** Can a learned matching or relocalization component improve a fixed localization pipeline under visual degradation and sensor misalignment?

**Build: baseline.** Build a reproducible classical baseline from existing components, with a fixed EKF. Use Oxford RobotCar if the [data audit](reports/data-audit.md) confirms access and terms, with one traverse as the reference map and traverses in other conditions as queries. KITTI is the fallback. Establish reference-pose quality first.

**Build: learned measurement.** Replace one component: learned feature matching or place recognition. Keep the estimator fixed, so any gain comes from the measurement.

**Build: robustness.** Evaluate blur, visual dropout, time offsets and odometry corruption. Evaluate real changes in weather and light between traverses as well. Keep tuning and evaluation traverses separate.

**Build: learned trust.** Cache visual features and train a small model with temporal positions and one cross-attention block. It uses a visual and odometry window to set how much the EKF trusts each visual measurement. Drop this step if the baseline is not working by week 16.

**Comparison.** All versions use the same traverses and tests:

| Version | Vision input | State estimator | Trust in vision |
| ------- | ------------ | --------------- | --------------- |
| Classical | Classical matching | EKF | Fixed |
| Learned measurement | Learned component | EKF | Fixed |
| Learned trust | Learned component | EKF | Predicted by cross-attention |

**Metrics.** Pose error against stated reference poses, by condition; lost-localization rate; recovery time; behavior under each corruption; runtime per frame.

**Completion criteria.** Compare all versions with the same protocol. Explain where each helps, fails or learns shortcuts. Save reproducible configs and a standalone report.

**Scope and interview defense.** No new data collection. Label injected odometry faults, delay and dropout as simulated, and describe reference poses by how they were produced. These results come from a road car, so they do not show forklift performance, and they are not a driving safety function. A factor graph and a particle filter are extensions. Defend time alignment, reference quality, noise models and what the system trusts.

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

**Question.** How does pose quality affect novel-view quality on one public sequence?

**Build.** Run Structure from Motion and small-scale bundle adjustment, train 3D Gaussian Splatting, and compare pose sources on held-out views.

**Limit.** Good novel views do not prove accurate metric geometry or useful data augmentation, so make neither claim.
