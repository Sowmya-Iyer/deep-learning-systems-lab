# Data audit

Week 3, bounded to a few hours. Confirm these facts from each dataset's own page before writing model code. The notes are starting points, not verified facts.

| Dataset | Used for | Check first | Notes so far | License | Size | Decision |
| ------- | -------- | ----------- | ------------ | ------- | ---: | -------- |
| [LOCO](https://github.com/tum-fml/loco) | Projects 1, 2 | Crop counts after splitting, split unit, usable crop size | 37,988 images, 5,593 labeled, 5 environments. Details in [LOCO audit](#loco-audit) | CC0 1.0 | 37,988 images; GB to confirm | Keep, with four classes and a custom split |
| [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) | Project 3, first choice | Is download access open? Which traverses have [RTK ground truth](https://arxiv.org/abs/2002.10152)? What odometry can the EKF use? Which few traverses cover different conditions? | Registration required. RTK release terms still to confirm | CC BY-NC-SA 4.0 per the site | — | — |
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Project 3, fallback | What are the license and official download sizes? Do the raw recordings supply usable IMU data? | Poses are published for sequences 00–10. Less change in weather and light, so it answers the condition question less well | — | — | — |
| [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) | Projects 1, 2, fallback | What do the official terms say? Sources disagree | Use only if LOCO fails the rule below | — | — | — |
| [nuImages](https://www.nuscenes.org/nuimages) | Projects 1, 2, fallback | What are the license and download size? | Use only if LOCO fails the rule below | — | — | — |
| [SimReady Warehouse](https://hyper.ai/en/datasets/nvidia-physical-ai-simready-warehouse-openusd) | Project 4 | Can it render readable labels? | Synthetic 3D assets | — | — | — |

## Decisions

- **Project 1 and 2 classes and split unit:** Pallet, small load carrier, stillage and pallet truck, split by LOCO subset using [loco_split.json](loco_split.json). Forklift is an exploratory result only.
- **LOCO, keep or replace:** Keep. All four classes pass the rule with this split.
- **Project 3 dataset:** —
- **Project 3 reference-pose description:** —
- **Odometry source:** real or simulated, with reason —
- **Project 4 synthetic label source:** —
- **Project 4 real-image holdout source:** —

**LOCO rule.** Keep LOCO if at least three useful classes each have 500 training crops and 100 test crops after splitting. That leaves 50 examples per class at the 10% label budget. Otherwise, replace it with BDD100K or nuImages for Projects 1 and 2, and rename those projects to match. Decide by Nov 8.

Download only a few RobotCar traverses, and record their total size here.

If no dataset has usable odometry, Project 3 simulates it from reference poses. Record the noise model here and in the report.

## LOCO audit

[audit_loco.py](audit_loco.py) counts candidate crops from the annotation files. It opens no images, so these are box counts, not checked crops. Run it from `reports/`, with the LOCO repository cloned as `loco/`.

```bash
python audit_loco.py --repo ../loco --min-side 32 --split-file loco_split.json --out loco_audit
```

### Annotation totals

| Source | Images | Boxes |
| ------ | -----: | ----: |
| LOCO repository, as listed | 5,593 labeled | 152,421 |
| LOCO class table, summed | — | 151,068 |
| Annotation files, as counted | 5,097 | 151,428 |

The three sources disagree, and the cause is unknown. This audit uses the annotation files. The 496 image difference also changes the unlabeled count for Project 2, so check that before pretraining.

### Filtering

Keeping boxes with a shorter side of at least 32 pixels retains 86,599 of 151,428, or 57%. There were no invalid boxes, and no crowd annotations among those kept.

### Crops by subset

| Subset | Images | Pallet | Small load carrier | Stillage | Pallet truck | Forklift |
| ------ | -----: | -----: | -----------------: | -------: | -----------: | -------: |
| 1 | 1,075 | 6,967 | 4,646 | 1,695 | 533 | 5 |
| 2 | 1,070 | 13,489 | 1,774 | 2,325 | 209 | 138 |
| 3 | 873 | 16,217 | 1,355 | 335 | 253 | 187 |
| 4 | 1,202 | 24,548 | 0 | 28 | 542 | 114 |
| 5 | 877 | 5,545 | 4,493 | 1 | 1,061 | 139 |

### Split choice

The first split (train 2, 3, 5; validation 1; test 4) fails the LOCO rule. Test has no small load carriers and only 28 stillages, so only pallet and pallet truck pass. Subsets 4 and 5 each lack a main class, so both belong in training.

The chosen split is train 2, 4, 5; validation 1; test 3. Subsets 1 and 3 were chosen from these counts alone, before any model was trained. Freeze it now, and do not change it after seeing results.

| Split | Pallet | Small load carrier | Stillage | Pallet truck | Forklift |
| ----- | -----: | -----------------: | -------: | -----------: | -------: |
| Train | 43,582 | 6,267 | 2,354 | 1,812 | 391 |
| Validation | 6,967 | 4,646 | 1,695 | 533 | 5 |
| Test | 16,217 | 1,355 | 335 | 253 | 187 |

The smallest main class has 181 training crops at the 10% label budget. Forklift fails the rule in every split tested, and has only 5 validation crops here.

### Limits

- **Test difficulty.** Test comes from a subset the model never trained on, so scores may be lower than with a random split. Name it a held-out subset result, and confirm what each subset contains before calling it a new site.
- **Imbalance.** Pallets are 80% of training crops, so report macro-F1 and per-class recall, and consider capping pallet crops.
- **Duplicates.** Source paths do not overlap between subsets, but visual duplicates were not checked.
- **Unfinished checks.** No crops have been extracted or viewed, and sequence identity within a subset is unknown. Do not split within a subset without it.
