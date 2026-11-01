# video-sensor

Learned localization, factor-graph estimation and multimodal fusion for [Project 3](../../PROJECTS.md). Week 13 setup, weeks 14-16 bounded experiments, week 18 report.

## Layout

```
video-sensor/
├── alignment/      # timestamps, sync and split code
├── classical/      # ORB reference matching and EKF baseline
├── learned/        # place recognition and feature matching
├── graph/          # sliding-window factor graph with GTSAM
├── fusion/         # cross-attention reliability model on cached features
├── experiments/    # blur, low texture, wheel slip, dropouts, time offsets
└── configs/
```

## Protocol

- Use the dataset chosen in the [data audit](../../reports/data-audit.md). No new data collection.
- Rebuild the classical baseline from published methods. Do not use employer code or parameters.
- Change one thing at a time: vision input, then estimator, then learned trust.
- Run all four versions on the same routes and tests.
- Keep tuning and evaluation route segments separate, and train the fusion model only on tuning segments.
- If no usable odometry exists, simulate it from ground-truth poses with documented noise.

| Version | Vision input | State estimator | Trust in vision |
| ------- | ------------ | --------------- | --------------- |
| Classical | ORB | EKF | Fixed |
| Learned front end | Learned matching | EKF | Fixed |
| Upgraded back end | Learned matching | Factor graph | Fixed, with outlier rejection |
| Full system | Learned matching | Factor graph | Predicted by cross-attention |

## Done when

- [ ] Camera and odometry timestamps are aligned and checked
- [ ] All four versions are compared with the same protocol
- [ ] Repeated-aisle false matches are tested directly
- [ ] Robustness is measured under blur, low texture, wheel slip, dropouts and time offsets
- [ ] Pose error, lost-localization rate, recovery time and runtime are reported
- [ ] The report explains what each change contributed, and where it helps, fails or learns shortcuts

An EKF beside the graph for fast output, and a particle filter for global localization, are extensions.

Results go in the [Project 3 report](../../reports/project-3-video-sensor/report.md).
