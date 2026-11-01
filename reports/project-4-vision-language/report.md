# Project 4: Aisle and Rack Label Reading

**Research question.** Can LoRA adaptation read aisle and rack labels better than OCR and a pretrained VLM while preserving quality on unseen strings, layouts and real images?

## Summary

The answer to the research question, with the main numbers.

## Protocol

Dataset, splits, models compared, and what was held constant.

## Environment

| Hardware | Precision | Batch size | Training budget | Seeds |
| -------- | --------- | ---------: | --------------: | ----: |
| — | — | — | — | — |

## Results

| Config | Field exact match | Record exact match | Character error rate | Schema validity | Abstention quality | Hallucinated labels | Real-image holdout | Peak memory |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| OCR + parsing | — | — | — | — | — | — | — | — |
| Pretrained VLM | — | — | — | — | — | — | — | — |
| LoRA VLM | — | — | — | — | — | — | — | — |

## Analysis

What improved, what regressed, and why. Include failures and negative results.

## Limitations

Compute limits, threats to validity, and what these results do not show.

## Reproduction

Commands and configs that regenerate the tables above.

## Defense

Questions to answer without notes:

- How do you know the model uses the image?
- Why is valid JSON not a correct reading?
- What did adaptation regress on unseen data?
- What do synthetic-only results not show?
- Why is a hallucinated label a localization failure?
