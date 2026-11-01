# vision-language

Aisle and rack label reading with SmolVLM for [Project 4](../../PROJECTS.md). Week 13 baseline, week 16 adaptation and report.

## Layout

```
vision-language/
├── preparation/    # synthetic label generation, image-text examples, split checks
├── baseline/       # OCR baseline and pretrained SmolVLM, with fixed prompts and decoding
├── adaptation/     # LoRA training
├── evaluation/     # exact match, general holdout, visual dependence, hallucination audit
└── configs/
```

## Protocol

- Target output is structured JSON such as `{aisle, bay, level}`.
- Use synthetic labels or images with permission. Check model and data licenses, duplicates and split integrity before training.
- Fix prompts, preprocessing, decoding and evaluation before any adaptation.
- Test fine-tuning memory on the laptop early. Rent a GPU if it does not fit in 8 GB.
- Before a full run, verify trainable parameters and loss masking, and overfit a tiny batch.

## Done when

- [ ] The OCR baseline, pretrained model and one adaptation setting are compared
- [ ] Removing or shuffling images tests visual dependence
- [ ] A separate general holdout is evaluated and hallucinated labels are counted
- [ ] Trainable parameter count, peak memory and training time are reported
- [ ] Adapter checkpoint or reproduction instructions are available

Training a VLM from scratch is outside scope. Feeding aisle readings into the Project 3 factor graph is an extension.

Results go in the [Project 4 report](../../reports/project-4-vision-language/report.md).
