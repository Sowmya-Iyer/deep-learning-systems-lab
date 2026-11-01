# 05-self-supervised

Label-efficient representation learning on warehouse images for [Project 2](../PROJECTS.md). Weeks 9-12, on a rented GPU.

## Layout

```
05-self-supervised/
├── splits/         # frozen train/validation/test and 10%, 25%, 100% label subsets, tracked in Git
├── baselines/      # supervised training at each label budget
├── pretrain/       # one contrastive encoder, trained on training images only
├── evaluation/     # linear probe, fine-tuning, existing pretrained encoder
├── ablations/      # one or two factors, such as augmentation strength or duration
└── configs/
```

## Protocol

1. Freeze the splits and label subsets before any training.
2. Train supervised baselines at 10%, 25% and 100% of labels.
3. Pretrain on training images without labels. Validation and test images stay out unless the report explicitly defines a transductive study.
4. Linear-probe the frozen encoder, then fine-tune it, using the same downstream splits and evaluation rules throughout.
5. Compare against an existing pretrained visual encoder.
6. Use at least 3 seeds for key comparisons when feasible, and record any compute limitation.

## Done when

- [ ] Accuracy or macro-F1 versus label budget is reported with variation across seeds
- [ ] Pretraining cost and class-level errors are reported
- [ ] The report states whether pretraining helped, by how much and under which conditions

Complete one method deeply. MAE, DINO training and auto-labeling are extensions.

Results go in the [Project 2 report](../reports/project-2-label-efficient/report.md).
