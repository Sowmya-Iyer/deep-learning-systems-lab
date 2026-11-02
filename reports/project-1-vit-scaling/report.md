# Project 1: Warehouse Recognition & ViT Scaling

**Research question.** How do CNN and ViT training choices affect warehouse object classification, generalization and training cost?

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | Macro-F1 | Lowest per-class recall | Accuracy | Training time | Samples/second | Peak GPU memory | Time to target | Cost |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- Why is this crop classification and not detection?
- How did splitting before cropping prevent leakage?
- What made training unstable, and how was it diagnosed?
- Where was the bottleneck, and what evidence shows it?
- What numerical behavior did AMP introduce?
- Why these distributed choices?
