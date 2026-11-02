# Data audit

Week 3, bounded to a few hours. Confirm these facts from each dataset's own page before writing model code. The notes are starting points, not verified facts.

| Dataset | Used for | Check first | Notes so far | License | Size | Decision |
| ------- | -------- | ----------- | ------------ | ------- | ---: | -------- |
| [LOCO](https://github.com/tum-fml/loco) | Projects 1, 2 | Class counts, split unit, unannotated images | Forklift annotations are few, so audit before choosing classes | — | — | — |
| [TorWIC](https://arxiv.org/pdf/2205.01202) | Project 3 | Is reference-pose quality usable? | Has RGB-D, wheel encoder and IMU. Reference poses come from proprietary LiDAR SLAM | — | — | — |
| [TorWIC extension](https://arxiv.org/pdf/2307.00488) | Project 3 | Is wheel odometry included? | Its authors report jumps and drifts in the original ground truth; this release uses a multistation scan | — | — | — |
| [Fraunhofer Industry](https://arxiv.org/pdf/1912.08263) | Project 3 | Are synchronized odometry or IMU streams available? | Forklift-mounted cameras with precise labeled poses | — | — | — |
| [SimReady Warehouse](https://hyper.ai/en/datasets/nvidia-physical-ai-simready-warehouse-openusd) | Project 4 | Can it render readable labels? | Synthetic 3D assets | — | — | — |

## Decisions

- **Project 1 and 2 classes and split unit:** —
- **Project 3 dataset:** —
- **Project 3 reference-pose description:** —
- **Odometry source:** real or simulated, with reason —
- **Project 4 synthetic label source:** —
- **Project 4 real-image holdout source:** —

If no dataset has usable odometry, Project 3 simulates it from reference poses. Record the noise model here and in the report.
