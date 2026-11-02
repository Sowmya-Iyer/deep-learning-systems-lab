# 05-self-supervised

Label-efficient representation learning on warehouse images for [Project 2](../PROJECTS.md). Weeks 9-12, on a rented GPU.

## Layout

```
05-self-supervised/
├── splits/         # frozen splits and nested 10%, 25%, 100% label subsets, tracked in Git
├── baselines/      # scratch training and external pretrained fine-tuning at each budget
├── pretrain/       # one contrastive encoder, trained on training images only
├── evaluation/     # linear probe and fine-tuning
├── ablations/      # one or two factors, such as augmentation strength or duration
└── configs/
```

## Protocol

1. Freeze the splits and nested, class-stratified label subsets before any training.
2. At 10%, 25% and 100% of labels, train from scratch and fine-tune an externally pretrained encoder.
3. Pretrain on training images without labels. Validation and test images stay out unless the report explicitly defines a transductive study.
4. Linear-probe the frozen encoder, then fine-tune it, using the same downstream splits and evaluation rules throughout.
5. Use at least 3 seeds for key comparisons when feasible, and record any compute limitation.

## Done when

- [ ] Macro-F1 versus label budget is reported with variation across seeds
- [ ] All three approaches are compared, because beating scratch training alone does not show SSL is useful
- [ ] Pretraining cost and class-level errors are reported
- [ ] The report states whether pretraining helped, by how much and under which conditions

Label efficiency and new-site adaptation are separate claims; make the second only with a held-out site. Complete one method deeply. MAE, DINO training and auto-labeling are extensions.

Results go in the [Project 2 report](../reports/project-2-label-efficient/report.md).
