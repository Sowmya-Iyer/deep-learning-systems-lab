# vision-language

Aisle and rack label reading for [Project 4](../../PROJECTS.md). Week 13 baseline, week 16 adaptation, week 17 evaluation, week 18 report.

## Layout

```
vision-language/
├── preparation/    # synthetic label generation, image-text examples, split checks
├── baseline/       # OCR with deterministic parsing, and pretrained SmolVLM
├── adaptation/     # LoRA training
├── evaluation/     # field and record match, holdouts, abstention, hallucination audit
└── configs/
```

## Protocol

- Target output is structured `{aisle, bay, level}` fields.
- Use synthetic labels or images with permission. Check model and data licenses, duplicates and split integrity before training.
- Fix prompts, preprocessing, decoding and evaluation before any adaptation.
- Test fine-tuning memory on the laptop early. Rent a GPU if it does not fit in 8 GB.
- Before a full run, verify trainable parameters and loss masking, and overfit a tiny batch.
- Keep a small, fixed, separately sourced real-image holdout. Synthetic-only results are not warehouse-readiness claims.

## Done when

- [ ] OCR, the pretrained VLM and LoRA are compared
- [ ] Field-level match, complete-record match and character error rate are reported
- [ ] Schema validity is scored separately from content accuracy
- [ ] Missing and unreadable labels test appropriate abstention
- [ ] Unseen strings, fonts, layouts and capture conditions are tested
- [ ] Removing or shuffling images tests visual dependence, and hallucinated labels are counted
- [ ] Trainable parameter count, peak memory and training time are reported
- [ ] Adapter checkpoint or reproduction instructions are available

Training a VLM from scratch is outside scope. Feeding aisle readings into Project 3 is an extension.

Results go in the [Project 4 report](../../reports/project-4-vision-language/report.md).
