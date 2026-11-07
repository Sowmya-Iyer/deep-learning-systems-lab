# Project 1 plan

Step-by-step work for [Project 1](../../PROJECTS.md#project-1-warehouse-detection--vit-scaling), in order. Each step ends in something checkable. Results go in the [report](report.md); this file only tracks the work.

**Fixed inputs.** The split is frozen in [loco_split.json](../loco_split.json): train subsets 2, 4, 5 (3,149 images); validation subset 1 (1,075); test subset 3 (873). Five classes. Forklift is reported with a warning and never used for tuning or model selection.

**Test rule.** Tune and select only on validation. Each model is scored on test once, after its config is final, and that number is recorded even if it is bad.

**Cut order if the schedule slips.** DETR comparison first, then extra seeds, then four GPUs and FSDP. Never cut evaluation.

---

## Phase 0: Data on disk (week 3, by Nov 8)

Only the annotation files are in `loco/` now. No images have been downloaded.

1. **Download the annotated images** from the link in the LOCO README (`go.mytum.de/239870`). Not the unannotated extra data. Put them outside OneDrive, under `DATA_ROOT/loco/`, and record the download size in the [data audit](../data-audit.md).
2. **Resolve every path.** Write `reports/check_loco_images.py`: for each image in the five subset JSONs, find the file from its `path` field (`/dataset/subset-N/<file_name>`), open it, and check that its size matches the JSON `width` and `height`. Report missing, unreadable and mismatched files per subset. Expected total: 5,097.
3. **Record image sizes per split.** Sizes are mixed: 1920×1080 everywhere, plus 1280×720/800/960 in subsets 2–5 and 640×480 only in subset 1 (validation). Count each size per split. If many validation images are 640×480, validation is harder than train for small objects, so note it in the audit.
4. **Look at the boxes.** Write `reports/show_loco_boxes.py` to draw boxes on 10 random images per subset. Confirm `bbox` is `[x, y, w, h]` in pixels and lines up with objects.
5. **Unlabeled-object check.** On 50 random validation images, count clearly visible objects of the five classes that have no box. Record the rate per image in the audit. This is the correction for false positives per image later.
6. **Install the missing tools** into `.torch-venv` and pin them in `01-pytorch-fundamentals/requirements.txt`: `pycocotools` and `pyyaml`. Install the package with `pip install -e training-pipeline`.

**Done when** every image resolves, the size table and unlabeled rate are in the data audit, and the audit's "GB to confirm" is filled in.

---

## Phase 1: The `dlsl` pipeline (week 3–4)

Build the modules listed in the [training-pipeline README](../../01-pytorch-fundamentals/training-pipeline/README.md), each with a test in `training-pipeline/tests/`.

1. **`seed.py`.** One `seed_everything(seed, deterministic)` for Python, NumPy and torch, plus DataLoader worker seeding. Test: two runs of 5 steps with the same seed give the same loss.
2. **`config.py`.** Load a YAML config, apply command-line overrides, resolve `DATA_ROOT`, and save the resolved copy to the run directory. Test: a saved config reloads to the same values.
3. **`data.py`.**
   - `LocoDetection`: reads the subset JSONs listed by `loco_split.json` for one split. Map LOCO category IDs `3, 5, 7, 10, 11` to contiguous labels `1–5` (0 is background) and keep the map in one place. Convert `xywh` to `xyxy`, drop zero-area boxes, and return torchvision-style targets with `image_id`, `area` and `iscrowd`.
   - Transforms that move boxes with the image: horizontal flip, and resizing left to the model.
   - `collate_fn` that returns lists, since images differ in size.
   - Tests: a hand-made image with one box survives conversion and flipping with the right coordinates; label IDs are 1–5; split images do not overlap.
4. **`metrics.py`.** COCO evaluation with pycocotools on the original image coordinates:
   - mAP@50, mAP@50:95, per-class AP, AR@100 as recall, and AP for small objects.
   - COCO "small" means area under 32² pixels, while the audit filtered by shorter side under 32. Report COCO small AP and say which definition it is.
   - False positives per image: predictions with score ≥ 0.5 that match no ground truth at IoU 0.5, per class and overall.
   - **Model-selection metric:** mean mAP@50:95 over the four main classes, without forklift.
   - Tests: perfect predictions give AP 1.0; one shifted box gives the hand-computed result; empty predictions give 0.
5. **`checkpoint.py`.** Save model, optimizer, scheduler, AMP scaler, all RNG states, epoch, global step and best metric. Write to a temporary file and rename, so an interruption never leaves a corrupt checkpoint. Keep `last.pt` and `best.pt`. Test: save, reload into fresh objects, and the next step's loss matches the uninterrupted run.
6. **`logging.py`.** JSONL per step and per epoch: losses, learning rate, images/second, peak GPU memory. Write an `environment.json` at the start: GPU, driver, torch and CUDA versions, git commit, precision, batch size, budget and seed.
7. **`engine.py`.** `train_one_epoch` with AMP on/off, gradient accumulation, gradient clipping, and a stop on non-finite loss that names the step. `evaluate` returns the metrics dict.
8. **Entry points** in `training-pipeline/`: `train.py --config ... [--resume path]` and `evaluate.py --checkpoint ... --split validation|test`.
9. **Overfit check.** Train on 16 training images for a few hundred steps. Loss should approach zero and validation-on-those-16 mAP@50 should approach 1. If not, the pipeline is broken; fix it before Phase 2.
10. **Resume check.** Run 2 epochs on a small subset uninterrupted. Run again, kill it partway through epoch 2, resume. The resumed loss curve and final metrics should match. Record the tolerance; cuDNN is not fully deterministic, so state what "match" means.

**Done when** all tests pass, the 16-image overfit works, resume matches, and `evaluate.py` on a saved checkpoint reproduces the logged validation numbers.

---

## Phase 2: Faster R-CNN baseline and failure analysis (week 4, Nov 9–15)

1. **Model.** torchvision `fasterrcnn_resnet50_fpn_v2` with COCO weights. Replace the box predictor with 6 outputs (5 classes plus background). Config: `configs/frcnn_r50_baseline.yaml`.
2. **Resolution.** The default resize is shorter side 800. That shrinks 1920×1080 images by about 0.74 and makes the 43% of boxes already under 32 pixels even smaller. Run one short comparison of shorter side 800 against 1080 on validation, and keep whichever fits in 8 GB with AMP and does better. Freeze the choice.
3. **Time one epoch** on the laptop with AMP and batch size 2. From it, estimate the full run, decide laptop versus Colab or a rented GPU, and update the compute budget in the [README](../../README.md#compute).
4. **Train the baseline.** SGD with momentum, warmup, then step or cosine decay, about 12–20 epochs as the timing allows. Select `best.pt` by the selection metric on validation. Run 3 seeds if the epoch time allows; otherwise 1, and record why.
5. **Evaluate once on test.** Fill the Environment and first Results row in the [report](report.md).
6. **Failure analysis on validation.**
   - Per-class AP, and small versus medium versus large.
   - False positives per image, before and after correcting with the Phase 0 unlabeled rate.
   - Confusions between classes, especially pallet against small load carrier and stillage.
   - Save image galleries of the worst misses and the highest-scoring false positives, then write down what they have in common.
   - Results by image size, since validation has 640×480 images that training has not seen.
7. **Fresh-clone check.** Clone into a new folder, follow only the README, train for 1 epoch, and evaluate the committed config. Fix any step that needed knowledge outside the README.

**Done when** the baseline row is in the report, the failure analysis is written up, and a fresh clone reproduces it. This is also the week 4 roadmap milestone.

---

## Phase 3: Attention, transformer blocks and a small ViT on crops (weeks 5–6, Nov 16–29)

Code lives in [03-transformers](../../03-transformers/).

1. **Attention.** Scaled dot-product attention and multi-head attention from scratch. Tests: output shapes, masking (masked positions get zero weight), and a match with `torch.nn.functional.scaled_dot_product_attention` within tolerance.
2. **Transformer block.** Pre-norm block with MLP and residuals. Test shapes and that gradients reach every parameter.
3. **Patch embedding and ViT.** Patch embedding, class token, learned positions, a stack of blocks and a classifier head. Start small, around 4–6 blocks at width 192.
4. **Crop dataset.** Extract crops with shorter side at least 32 from the frozen split, using the audit's crop manifest. Resize to one fixed size, such as 64 or 96. Cap pallet crops in training, since pallets are 80% of crops. Spot-check 50 crops by eye.
5. **Tiny-batch overfit**, then train on crops. Report accuracy, macro-F1 and per-class recall.
6. **Baseline on the same crops.** A small CNN (the existing `smallcnn`) and a pretrained ResNet-18, with the same budget and evaluation. The ViT need not win; explain the gap.

**Done when** the tests pass and the crop table is in the report, labeled as crop classification, not detection.

---

## Phase 4: Detector comparison and training measurements (weeks 7–8, Nov 30–Dec 13)

1. **DETR-family detector.** Fine-tune one pretrained DETR-family model from COCO weights, such as RT-DETR or Deformable DETR. Check its license first. Keep the split, input resolution, evaluation code and GPU-hour budget the same as Faster R-CNN. Present it as a transfer-learning study, since backbone, pretraining and training recipe all differ. Drop this step first if the schedule slips.
2. **AMP.** Faster R-CNN with and without AMP: images/second, peak memory, final mAP, and how often the gradient scaler skipped a step.
3. **Batch size.** Batch 1, 2 and 4 if it fits, with the learning rate scaled. Record throughput, memory and mAP.
4. **Data loading.** `num_workers` 0, 2, 4, 8 and `pin_memory` on/off. Use `torch.profiler` to show whether the GPU waits on data. Decoding 1920×1080 JPEGs is a likely bottleneck.
5. **Gradient accumulation.** Batch 4 against batch 2 with 2 accumulation steps. Losses should nearly match. The torchvision detector freezes BatchNorm, so say whether that holds in your config and why it matters.
6. Record hardware, precision, batch size, budget and seed for every row.

**Done when** the first comparison and measurement tables are in the report.

---

## Phase 5: DDP and memory (weeks 19–20, Feb 15–28)

Follow [04-distributed](../../04-distributed/README.md).

1. On the laptop, run DDP with `gloo` and two CPU processes on a tiny subset.
2. Test that `DistributedSampler` shards are disjoint, cover the dataset, and reshuffle when `set_epoch` is called.
3. Save checkpoints from rank 0 only, from the unwrapped model, and resume under DDP.
4. Rent one two-GPU machine and measure 1 GPU against 2 GPUs: throughput, time to target, peak memory, scaling efficiency and cost.
5. Measure activation checkpointing on the backbone: memory saved against speed lost. Use FSDP only if the memory results justify it.

**Done when** the 04-distributed table is filled from measured runs.

---

## Phase 6: Report

Fill every section of the [report](report.md): summary, protocol, environment, results, analysis, limitations and reproduction commands. Answer each defense question from your own runs. Then tick the "Done when" boxes in the training-pipeline, 03-transformers and 04-distributed READMEs, and the README progress table.
