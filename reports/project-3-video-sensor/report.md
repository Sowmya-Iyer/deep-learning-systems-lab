# Project 3: Learned Forklift Localization

**Research question.** When do learned place recognition and matching improve on ORB + EKF, and what happens when vision or odometry is degraded, missing or misaligned?

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | Pose error | Lost-localization rate | Recovery time | Blur | Wheel slip | Dropouts | Time offset | Runtime per frame |
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

- How were camera and odometry aligned, and how was that checked?
- How was measurement covariance chosen for learned matches?
- When does the filter trust vision, and when odometry?
- Where does the learned front end help, fail or learn shortcuts?
