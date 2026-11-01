# Project 3: Learned Forklift Localization

**Research question.** When do learned measurements and a factor graph improve on ORB + EKF, and what happens when vision or odometry is degraded, missing or misaligned?

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
| Classical: ORB + EKF | — | — | — | — | — | — | — | — |
| Learned front end: learned matching + EKF | — | — | — | — | — | — | — | — |
| Upgraded back end: learned matching + factor graph | — | — | — | — | — | — | — | — |
| Full system: factor graph + predicted trust | — | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- How were camera and odometry aligned, and how was that checked?
- Why does an EKF struggle with repeated aisles, and did the factor graph fix that?
- How does predicted reliability become a noise model, and is it calibrated?
- When does the system trust vision, and when odometry?
- What did each change contribute, and where did it fail or learn shortcuts?
