# video-sensor

Learned visual measurements and trust modeling for [Project 3](../../PROJECTS.md). Week 13 setup, weeks 14-15 experiments, week 17 trust model, week 18 report.

## Layout

```
video-sensor/
├── alignment/      # timestamps, sync and split code
├── classical/      # baseline from existing components, with a fixed EKF
├── learned/        # one learned matching or place-recognition component
├── trust/          # cross-attention trust model on cached features
├── experiments/    # blur, visual dropout, time offsets, odometry corruption
└── configs/
```

## Protocol

- Use the dataset chosen in the [data audit](../../reports/data-audit.md). No new data collection.
- Establish reference-pose quality first, and describe reference poses by how they were produced.
- Use existing components for the baseline. Do not use employer code or parameters.
- Keep the EKF fixed, and change one thing at a time.
- Keep tuning and evaluation route segments separate, and train the trust model only on tuning segments.
- Label injected wheel slip, delay and dropout as simulated.

| Version | Vision input | State estimator | Trust in vision |
| ------- | ------------ | --------------- | --------------- |
| Classical | Classical matching | EKF | Fixed |
| Learned measurement | Learned component | EKF | Fixed |
| Learned trust | Learned component | EKF | Predicted by cross-attention |

## Done when

- [ ] Camera and odometry timestamps are aligned and checked
- [ ] Classical and learned versions are compared with the same protocol
- [ ] Robustness is measured under blur, visual dropout, time offsets and odometry corruption
- [ ] Pose error, lost-localization rate, recovery time and runtime are reported
- [ ] The trust model is compared with fixed trust, or dropped with a reason if the baseline was not working by week 16
- [ ] The report explains where each version helps, fails or learns shortcuts

A factor graph, a particle filter and fusing aisle labels are extensions.

Results go in the [Project 3 report](../../reports/project-3-video-sensor/report.md).
