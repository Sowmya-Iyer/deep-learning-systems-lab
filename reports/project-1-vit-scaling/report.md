# Project 1: Warehouse Detection & ViT Scaling

**Research question.** How do CNN and transformer training choices affect warehouse object detection, generalization and training cost?

The work is tracked step by step in the [plan](plan.md).

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | mAP@50 | mAP@50:95 | Lowest per-class AP | Recall | Small-object AP | False positives per image | Latency | Training time | Images/second | Peak GPU memory | Time to target | Cost |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- Why split by LOCO subset, and what does a held-out subset show?
- How do mAP, recall and false positives each describe a different failure?
- How do unlabeled objects affect the false-positive count?
- Why is the forklift result unreliable?
- What made training unstable, and how was it diagnosed?
- Where was the bottleneck, and what evidence shows it?
- What numerical behavior did AMP introduce?
- Why these distributed choices?
