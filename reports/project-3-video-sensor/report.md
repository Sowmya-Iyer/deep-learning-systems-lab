# Project 3: Robust Warehouse Localization with Learned Visual Measurements

**Research question.** Can a learned matching or relocalization component improve a fixed localization pipeline under visual degradation and sensor misalignment?

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | Pose error | Lost-localization rate | Recovery time | Blur | Visual dropout | Time offset | Odometry corruption | Runtime per frame |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + EKF | — | — | — | — | — | — | — | — |
| Learned measurement + EKF | — | — | — | — | — | — | — | — |
| Learned trust + EKF | — | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- How were camera and odometry aligned, and how was that checked?
- How were reference poses produced, and how far can they be trusted?
- Which results come from simulated corruption?
- How does predicted trust become a noise value, and is it calibrated?
- Where does each version help, fail or learn shortcuts?
