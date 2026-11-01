# 03-transformers

Attention, transformer blocks and a small ViT for [Project 1](../PROJECTS.md), trained on LOCO object crops with the shared [`dlsl`](../01-pytorch-fundamentals/training-pipeline/) pipeline. Weeks 5-8, on the laptop.

## Layout

```
03-transformers/
├── attention/      # scaled dot-product and multi-head attention, with tests
├── vit/            # transformer blocks, patch embeddings, small ViT
├── experiments/    # pretrained adaptation; AMP, batch size, data loading
│                   # and gradient accumulation measurements
└── configs/
```

## Done when

- [ ] Attention and multi-head attention have shape and masking tests
- [ ] The small ViT overfits a tiny batch, then trains on the same split as the CNN/ResNet baseline
- [ ] Pretrained CNN and ViT are compared separately, as a transfer-learning study
- [ ] Macro-F1, per-class recall and confusion matrices are reported
- [ ] AMP, batch size, data loading and gradient accumulation are each measured, with hardware, precision, batch size and training budget recorded

This is object-crop classification, not full-scene detection. The small ViT need not beat a pretrained model. The result must explain what improved and at what cost.

Results go in the [Project 1 report](../reports/project-1-vit-scaling/report.md).
