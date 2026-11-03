# Project 2: Label-Efficient Free-Space Segmentation

**Research question.** Does driving-domain self-supervised pretraining improve free-space segmentation over scratch training and an external pretrained encoder at 10%, 25% and 100% of training labels?

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | Label budget | Free-space IoU | Boundary accuracy | Free space inside objects | Night and bad-weather IoU | Variation across seeds | Pretraining cost |
| ------ | --- | --- | --- | --- | --- | --- | --- |
| Scratch | — | — | — | — | — | — | — |
| External pretrained | — | — | — | — | — | — | — |
| Driving SSL, frozen encoder | — | — | — | — | — | — | — |
| Driving SSL, fine-tuned | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- What was the hypothesis, and what result would have refuted it?
- Why is the external pretrained baseline necessary?
- What controls make the comparisons fair?
- Do these results show label efficiency, adaptation to new conditions, or both?
- Why is free-space IoU not enough on its own, and which errors matter most near obstacles?
- What negative results appeared, and how uncertain are the conclusions?
