# 03-transformers

Attention, transformer blocks, a small ViT and a DETR-family detector comparison for [Project 1](../PROJECTS.md), using the shared [`dlsl`](../01-pytorch-fundamentals/training-pipeline/) pipeline. Weeks 5-8, on the laptop first, with Colab if detector training needs it.

## Layout

```
03-transformers/
├── attention/      # scaled dot-product and multi-head attention, with tests
├── vit/            # transformer blocks, patch embeddings, small ViT
├── experiments/    # detector comparison; AMP, batch size, data loading
│                   # and gradient accumulation measurements
└── configs/
```

## Done when

- [ ] Attention and multi-head attention have shape and masking tests
- [ ] The small ViT overfits a tiny batch, then trains on object crops from the frozen split
- [ ] Faster R-CNN and a pretrained DETR-family detector are compared as a transfer-learning study
- [ ] mAP@50, mAP@50:95, per-class AP, recall and false positives per image are reported
- [ ] AMP, batch size, data loading and gradient accumulation are each measured, with hardware, precision, batch size and training budget recorded

The small ViT is a crop classifier that teaches the components, not a detector. It need not beat a pretrained model. The result must explain what improved and at what cost.

Results go in the [Project 1 report](../reports/project-1-vit-scaling/report.md).
