# video-sensor

Learned localization with odometry fusion for [Project 3](../../PROJECTS.md). Week 13 setup, weeks 14-15 bounded experiments, week 18 report.

## Layout

```
video-sensor/
├── alignment/      # timestamps, sync and split code
├── classical/      # ORB reference matching and EKF baseline
├── learned/        # place recognition and feature matching
├── experiments/    # blur, low texture, wheel slip, dropouts, time offsets
└── configs/
```

## Protocol

- Use the dataset chosen in the [data audit](../../reports/data-audit.md). No new data collection.
- Rebuild the classical baseline from published methods. Do not use employer code or parameters.
- Keep the same EKF for both front ends, and change only the measurement source.
- Keep tuning and evaluation route segments separate.
- If no usable odometry exists, simulate it from ground-truth poses with documented noise.

## Done when

- [ ] Camera and odometry timestamps are aligned and checked
- [ ] ORB + EKF and learned + EKF are compared with the same protocol
- [ ] Robustness is measured under blur, low texture, wheel slip, dropouts and time offsets
- [ ] Pose error, lost-localization rate, recovery time and runtime are reported
- [ ] The report explains where the learned front end helps, fails or learns shortcuts

Results go in the [Project 3 report](../../reports/project-3-video-sensor/report.md).
