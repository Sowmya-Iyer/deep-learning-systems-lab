# dlsl design

What each file in `src/dlsl/` contains, before it is written. Signatures are a starting point; change them when the code shows a better shape, and update this file when you do. The phases refer to the [Project 1 plan](../../reports/project-1-vit-scaling/plan.md).

```text
src/dlsl/
├── __init__.py
├── config.py
├── seed.py
├── checkpoint.py
├── runlog.py
├── optim.py
├── engine.py
├── metrics.py
├── train.py
├── evaluate.py
├── data/
│   ├── __init__.py
│   ├── loco.py
│   ├── transforms.py
│   └── crops.py
└── models/
    ├── __init__.py
    ├── detectors.py
    ├── attention.py
    ├── transformer.py
    └── vit.py
```

Build in this order: `seed`, `config`, `data/loco`, `data/transforms`, `models/detectors`, `metrics`, `checkpoint`, `runlog`, `optim`, `engine`, `train`, `evaluate`. That gives a working baseline before anything in Phase 3.

---

## A config, for reference

Every module reads from one nested dict loaded from YAML. This is roughly what `configs/frcnn_r50_baseline.yaml` will hold:

```yaml
experiment: frcnn_r50_baseline
seed: 0

data:
  annotations: ${REPO_ROOT}/loco/rgb        # the cloned LOCO repo, not committed
  images: ${DATA_ROOT}/loco                 # downloaded images, outside OneDrive
  split_file: ${REPO_ROOT}/reports/loco_split.json
  subset_size: null                         # e.g. 16 for the overfit check
  num_workers: 4
  pin_memory: true

model:
  name: fasterrcnn_resnet50_fpn_v2
  weights: COCO_V1
  num_classes: 6                            # 5 LOCO classes + background
  min_size: 800
  max_size: 1333
  trainable_backbone_layers: 3

optim: {name: sgd, lr: 0.01, momentum: 0.9, weight_decay: 1.0e-4}
schedule: {name: cosine, epochs: 16, warmup_iters: 500, warmup_factor: 0.001}

train:
  batch_size: 2
  accum_steps: 1
  amp: true
  clip_grad_norm: 10.0
  log_every: 20                             # steps
  checkpoint_every: 500                     # steps, for interruptible machines
  eval_every: 1                             # epochs

eval:
  score_threshold: 0.5                      # for false positives per image only
  iou_threshold: 0.5
  select_metric: map_main                   # mAP@50:95 over the four main classes

output:
  runs_root: ${RUNS_ROOT:-runs}
```

---

## `__init__.py`

- `__version__ = "0.1.0"`, kept in step with `pyproject.toml`.
- No other imports, so `import dlsl` stays fast and never pulls in torchvision.

## `config.py`

Load, override, resolve and save experiment configs.

- `load_config(path, overrides=()) -> dict`
  - Reads YAML with `yaml.safe_load`.
  - Applies overrides written as `train.batch_size=4`. Parse each value with `yaml.safe_load` so `4`, `true` and `1e-4` get the right types. An override for a key that does not exist is an error, which catches typos.
  - Resolves `${VAR}` and `${VAR:-default}` from environment variables. `REPO_ROOT` is set by the loader itself, found by walking up to the folder containing `.git`.
  - Fails with the key name if a required variable such as `DATA_ROOT` is unset.
- `save_config(cfg, run_dir)` writes `config.yaml`, the fully resolved copy.
- `get(cfg, "train.lr", default=None)` for dotted lookups.
- `config_hash(cfg) -> str`: a short hash of the resolved config, logged with every run.

Tests: overrides change values and types correctly; an unknown key raises; `${VAR:-default}` uses the default when unset; save then load gives the same dict.

## `seed.py`

Seeding and determinism in one place.

- `seed_everything(seed, deterministic=False)`
  - Seeds `random`, NumPy, `torch` and all CUDA devices.
  - With `deterministic=True`: `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark = False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. `warn_only` is needed because some torchvision detection ops, such as RoI Align backward on CUDA, have no deterministic version.
- `seed_worker(worker_id)` for `DataLoader(worker_init_fn=...)`: seeds NumPy and `random` from `torch.initial_seed()`.
- `make_generator(seed) -> torch.Generator` for `DataLoader(generator=...)`, so the shuffle order depends only on the seed.

Tests: two 5-step runs with the same seed give the same losses on CPU; different seeds differ.

## `data/__init__.py`

- `detection_collate(batch) -> (list[Tensor], list[dict])`. Images differ in size, so they stay a list; torchvision detectors accept that.
- `build_dataloader(dataset, cfg, split, seed) -> DataLoader`: shuffles only for train, and wires in `seed_worker`, `make_generator`, `num_workers` and `pin_memory`.
- Re-exports `LocoDetection` and `LocoCrops`.

## `data/loco.py`

Full-image LOCO detection on the frozen split.

- Class constants, defined once and used everywhere:
  - `LOCO_CATEGORIES = {3: "small_load_carrier", 5: "forklift", 7: "pallet", 10: "stillage", 11: "pallet_truck"}`
  - `LABEL_OF`: LOCO category ID to contiguous label 1–5, ordered by category ID. 0 is background.
  - `CLASS_NAMES`: label to name.
  - `MAIN_LABELS`: every label except forklift, used for model selection.
- `subsets_for(split_file, split) -> list[int]`: for example `train` gives `[2, 4, 5]`.
- `load_annotations(ann_dir, subsets) -> dict`: reads `loco-sub{N}-v1-*.json` for each subset and merges them.
  - Checks image IDs and annotation IDs are unique **across** subsets. The audit only checked within each file. If they collide, remap them and record that you did.
  - Keeps the subset number on each image for later breakdowns.
- `image_path(image_root, image) -> Path`: turns the JSON `path` (`/dataset/subset-N/<file_name>`) into a real file under `image_root`. Confirm the folder layout after the Phase 0 download and change only this function.
- `class LocoDetection(Dataset)`
  - `__init__(ann_dir, image_root, split_file, split, transforms=None, subset_size=None, seed=0)`. `subset_size` picks a fixed random set of images for the overfit check.
  - `__getitem__(i) -> (image, target)`:
    - `image`: float tensor `[3, H, W]` in `[0, 1]`, decoded with `torchvision.io.decode_image`.
    - `target`: `boxes` as float `[N, 4]` in `xyxy` pixels, `labels` as int64 1–5, `image_id`, `area`, `iscrowd`.
    - Converts `xywh` to `xyxy` and drops zero-area boxes.
  - `to_coco() -> dict`: ground truth in COCO format with contiguous category IDs 1–5, for `metrics.py`. Built from the same filtered boxes the model trains on.
  - `image_info(image_id) -> dict`: subset, file name, width and height, for breakdowns by image size.

Tests, using a tiny hand-written JSON in `tests/fixtures/`: label mapping is 1–5; `xywh` to `xyxy` is right; zero-area boxes are dropped; train, validation and test images do not overlap; `to_coco()` round-trips through pycocotools.

## `data/transforms.py`

Transforms that keep boxes aligned with the image. Resizing is left to the model, because torchvision detectors resize internally.

- `Compose(transforms)`: calls each with `(image, target)`.
- `RandomHorizontalFlip(p=0.5)`: flips the image and maps `x1, x2` to `W - x2, W - x1`.
- `ColorJitter(...)`: image only; boxes unchanged.
- `build_transforms(cfg, train: bool)`: flips only when training.

Tests: flipping twice returns the original boxes; a box at the left edge ends at the right edge; boxes stay inside the image.

## `data/crops.py`

Object crops for the Phase 3 classifiers. This is crop classification, not detection.

- `extract_crops(ann_dir, image_root, split_file, out_dir, min_side=32, context=0.1)`: one-off. Cuts each box with shorter side at least `min_side`, adds a little surrounding context, and writes PNGs under `out_dir/<split>/<class>/` with a `manifest.csv`. Decoding a 1920×1080 JPEG for every crop at training time would make the data loader the bottleneck, so crops are cached once.
- `class LocoCrops(Dataset)`
  - `__init__(crop_root, split, size=64, transforms=None, cap_per_class=None, seed=0)`
  - Returns `(image [3, size, size], label 0–4)`. Note that labels here start at 0, since there is no background class.
  - `cap_per_class` limits pallet crops in training.
  - `class_counts() -> dict` for the report.

Tests: every manifest row points to an existing file; capping never touches validation or test; labels are 0–4.

## `models/__init__.py`

- `build_model(cfg) -> nn.Module`: looks up `cfg["model"]["name"]` in one registry covering detectors and classifiers.

## `models/detectors.py`

- `build_fasterrcnn(num_classes, weights, min_size, max_size, trainable_backbone_layers) -> nn.Module`
  - torchvision `fasterrcnn_resnet50_fpn_v2` with COCO weights.
  - Replaces the box predictor with a `FastRCNNPredictor` of `num_classes` outputs.
  - Passes `min_size` and `max_size` through, for the 800 against 1080 comparison.
- Phase 4: `build_detr(...)` for the DETR-family model. It needs its own input processing and output conversion, so keep that conversion inside this file. Then `engine.py` always sees torchvision-style targets and predictions.

Test: the model built with `num_classes=6` returns boxes, labels and scores for a random image in eval mode, and a loss dict in train mode.

## `models/attention.py`

Phase 3, written from scratch.

- `scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0) -> (out, weights)`
  - Shapes: `q [B, H, Lq, d]`, `k, v [B, H, Lk, d]`.
  - Mask convention, same as PyTorch: a boolean mask where `True` means "may attend". Masked scores become `-inf` before the softmax.
  - Returns the weights too, for tests and plots.
- `class MultiHeadAttention(nn.Module)`
  - `__init__(dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)`
  - `forward(x, context=None, mask=None)`. With `context=None` it is self-attention; otherwise queries come from `x` and keys and values from `context`. That is the cross-attention Project 3 needs.

Tests: output shapes; masked positions get zero weight; weights sum to 1; matches `torch.nn.functional.scaled_dot_product_attention` within `1e-5` in float32; self-attention and cross-attention with `context=x` agree.

## `models/transformer.py`

- `class MLP(nn.Module)`: `Linear → GELU → Dropout → Linear → Dropout`, with hidden width `mlp_ratio * dim`.
- `class TransformerBlock(nn.Module)`: pre-norm, `x = x + attn(norm1(x))`, then `x = x + mlp(norm2(x))`.
- `class CrossAttentionBlock(nn.Module)`: the same with cross-attention to a `context` input. Written now, used in Project 3.

Tests: shapes are preserved; every parameter gets a gradient after one backward pass; with attention and MLP outputs zeroed, the block is the identity.

## `models/vit.py`

- `class PatchEmbed(nn.Module)`: `Conv2d(in_chans, dim, kernel_size=patch, stride=patch)`, then flatten to `[B, N, dim]`. Asserts the image size divides by the patch size.
- `class ViT(nn.Module)`
  - `__init__(img_size=64, patch_size=8, in_chans=3, num_classes=5, dim=192, depth=6, num_heads=3, mlp_ratio=4.0, dropout=0.0)`. 64-pixel crops with 8-pixel patches give 64 tokens.
  - Class token, learned position embedding, `depth` transformer blocks, final norm, linear head.
  - Initialization: truncated normal with std 0.02 for the position embedding, class token and linear layers.
  - `forward_features(x)` returns the class-token feature; `forward(x)` returns logits.
- `vit_tiny(**kwargs)`: a factory registered in `models/__init__.py`.

Tests: logits are `[B, 5]`; parameter count is in the expected range; the model overfits 8 random images to near-zero loss in a few hundred steps.

## `metrics.py`

COCO detection metrics, plus the counts COCO does not give.

- `coco_evaluate(gt: dict, predictions: list[dict]) -> dict`
  - `predictions` are COCO results: `image_id`, `category_id` 1–5, `bbox` as `xywh` in original image pixels, `score`.
  - Returns `map` (mAP@50:95), `map50`, `map75`, `map_small`, `map_medium`, `map_large`, `ar100`, `per_class_ap` as `{name: AP}`, and `map_main`, the mean AP over the four main classes.
  - Per-class AP comes from `COCOeval.eval["precision"]`, ignoring the `-1` entries for classes with no ground truth.
  - Hides pycocotools' printing, and returns zeros instead of crashing when there are no predictions.
  - COCO "small" means area under 32² pixels, not shorter side under 32 as in the audit. The returned dict names the definition so the report can quote it.
- `false_positives_per_image(gt, predictions, score_threshold=0.5, iou_threshold=0.5) -> dict`
  - Greedy matching per image and class, highest score first. A prediction is a false positive if it matches no unmatched ground-truth box.
  - Returns overall and per-class false positives per image, plus the image count.

Tests: perfect predictions give AP 1.0 and zero false positives; one prediction shifted to IoU 0.4 is a false positive at 0.5; empty predictions give zeros; a two-class case matches hand-computed AP@50.

## `checkpoint.py`

Save and resume everything a run needs to continue as if uninterrupted.

- `save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch, step, best_metric, config)`
  - Also saves RNG states: Python, NumPy, torch CPU and every CUDA device.
  - Saves from the unwrapped model: removes a DDP `.module` or a `torch.compile` `_orig_mod` first, so checkpoints load into a plain model.
  - Writes `path.tmp`, then `os.replace` onto `path`, so an interruption mid-write never corrupts `last.pt`.
- `load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True) -> dict`
  - Loads to CPU first with `map_location="cpu"`.
  - Recent PyTorch loads with `weights_only=True` by default, which rejects NumPy RNG state. Use `weights_only=False` only for checkpoints this pipeline wrote, and say so in a comment.
  - Returns `epoch`, `step`, `best_metric` and `config`.
- `find_last(run_dir) -> Path | None` for `--resume auto`.

Tests: save and load into fresh objects gives equal parameters, optimizer state and scheduler step; the next training step's loss matches an uninterrupted run on CPU; a leftover `.tmp` file never replaces `last.pt`.

## `runlog.py`

Named `runlog`, not `logging`, so it is never confused with the standard library module.

- `create_run_dir(cfg) -> Path`: `runs_root/<experiment>/<YYYYmmdd-HHMMSS>/`.
- `write_environment(run_dir, cfg)`: `environment.json` with GPU name and memory, driver, Python, torch, CUDA and cuDNN versions, OS, git commit and whether the tree had uncommitted changes, config hash, seed, precision, batch size and training budget. These are the facts every reported number must state.
- `class MetricLogger`
  - `log(kind, step, **values)` appends one JSON line to `metrics.jsonl`, with `kind` as `step`, `epoch` or `eval`, and flushes so an interruption loses nothing.
  - Also prints a short line through the standard `logging` module.
- `class Throughput`: counts images and wall time between log lines and reports images/second. Calls `torch.cuda.synchronize()` only at log points, so measuring does not slow training.
- `peak_memory_mb(reset=True)`: `torch.cuda.max_memory_allocated`, reset per epoch.

Tests: `metrics.jsonl` lines parse as JSON; `environment.json` has every required key; it works on CPU without CUDA.

## `optim.py`

- `build_optimizer(model, cfg) -> Optimizer`: SGD with momentum, or AdamW for the DETR-family model and the ViT. Only parameters with `requires_grad`. Optionally no weight decay on norm layers and biases.
- `build_scheduler(optimizer, cfg, iters_per_epoch) -> LRScheduler`: linear warmup, then cosine or step decay, stepped every **iteration**, so changing the batch size does not silently change the schedule.

Tests: the learning rate starts at `lr * warmup_factor`, reaches `lr` after warmup, and ends near zero for cosine.

## `engine.py`

The training and evaluation loops. They know nothing about LOCO, so later projects reuse them.

- `train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, cfg, logger, checkpoint_fn=None, skip_batches=0) -> int`
  - Mixed precision with `torch.autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp)` and `torch.amp.GradScaler`.
  - Gradient accumulation: divide the loss by `accum_steps`, and step only every `accum_steps` batches.
  - On each optimizer step: `scaler.unscale_`, clip the gradient norm, `scaler.step`, `scaler.update`, `zero_grad(set_to_none=True)`, `scheduler.step()`.
  - Counts steps the scaler skipped, for the AMP study in Phase 4.
  - Raises `NonFiniteLossError(step, loss_dict)` on a NaN or infinite loss, naming the loss term.
  - Logs every `log_every` steps: each loss term, learning rate, images/second, peak memory.
  - Calls `checkpoint_fn(step)` every `checkpoint_every` steps.
  - Mid-epoch resume: the epoch's shuffle order comes from `seed + epoch`, so resuming rebuilds the same order and skips `skip_batches`.
  - Returns the global step.
- `evaluate(model, loader, device, gt_coco, cfg) -> (metrics, predictions)`
  - `model.eval()` and `torch.inference_mode()`.
  - Converts outputs to COCO results in original coordinates, then calls `coco_evaluate` and `false_positives_per_image`.
  - Returns predictions too, so `evaluate.py` can save them for the analysis scripts.
- `measure_latency(model, images, device, warmup=10, iters=50) -> dict`: median and 90th-percentile milliseconds per image at batch size 1, with CUDA synchronization.

Tests on a tiny fake model and dataset on CPU: the loss goes down; `accum_steps=2` with batch 1 gives the same gradients as batch 2 for a model without batch-dependent layers; a NaN loss raises with the step number.

## `train.py`

Run with `python -m dlsl.train --config configs/frcnn_r50_baseline.yaml [--set key=value ...] [--resume PATH|auto]`.

1. Load the config with overrides. On resume, take the config from the checkpoint and refuse overrides that change the model or data.
2. Create the run directory, or reuse the resumed run's directory. Save the config and environment.
3. Seed everything.
4. Build the train and validation datasets and loaders. **Never the test set.**
5. Build the model, optimizer, scheduler and scaler.
6. On resume, load the checkpoint and work out the epoch and batches to skip.
7. For each epoch: train; evaluate on validation every `eval_every` epochs; save `last.pt`; copy to `best.pt` when `select_metric` improves.
8. Write `summary.json`: best epoch, best validation metrics, total training time, images/second, peak memory.

## `evaluate.py`

Run with `python -m dlsl.evaluate --checkpoint runs/.../best.pt --split validation|test`.

- Rebuilds the model and dataset from the config stored in the checkpoint, so evaluation never depends on a config file that may have changed.
- Writes `metrics_<split>.json` and `predictions_<split>.json` next to the checkpoint. The scripts in `training-pipeline/scripts/` read the predictions.
- Also measures latency, and records the hardware it ran on.
- **Test guard:** `--split test` requires `--confirm-test`, and appends a line to `runs/test_evaluations.jsonl` with the checkpoint, config hash and time. A config scored on test twice is then visible, which enforces the test-once rule.

Test: evaluating a saved checkpoint twice gives identical metrics, and they match the validation metrics logged during training.

---

## Later, not now

- `distributed.py` in Phase 5: process group setup from environment variables, `is_main_process()`, `barrier()`, wrapping the model in DDP, and `DistributedSampler` with `set_epoch`. `train.py` gains a flag for it; nothing above changes shape.
- `profiling.py` in Phase 4, if the `torch.profiler` setup grows beyond a few lines in the experiment scripts.
