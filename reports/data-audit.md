# Data audit

Weeks 1-2. Confirm these facts from each dataset's own page before writing model code. The notes below are starting points, not verified facts.

| Dataset | Used for | Check first | Sensors | Ground truth | License | Size | Decision |
| ------- | -------- | ----------- | ------- | ------------ | ------- | ---: | -------- |
| [LOCO](https://github.com/tum-fml/loco) | Projects 1, 2 | Forklift annotation count | — | — | — | — | — |
| [Fraunhofer Industry](https://arxiv.org/pdf/1912.08263) | Projects 3, 5 | Is odometry or IMU included? Is overlap enough for SfM? | — | — | — | — | — |
| [TorWIC warehouse extension](https://arxiv.org/pdf/2307.00488) | Projects 3, 5 | Sensor mount, odometry, IMU | — | — | — | — | — |
| [SDG-Warehouse](https://arxiv.org/pdf/2606.02800) | Project 5 extension | License, access, depth output | — | — | — | — | — |
| [PhysicalAI-SmartSpaces](https://docs.rcac.purdue.edu/datasets/ai/PhysicalAI-SmartSpaces/) | Project 5 extension | Forklift coverage | — | — | — | — | — |
| [SimReady Warehouse](https://hyper.ai/en/datasets/nvidia-physical-ai-simready-warehouse-openusd) | Projects 4, 5 | Can it render labels and depth? | — | — | — | — | — |

## Decisions

- **Project 3 dataset:** —
- **Odometry source:** real or simulated, with reason —
- **Project 4 label source:** —
- **Project 5 sequence:** —

If no dataset has usable odometry, Project 3 simulates it from ground-truth poses. Record the noise model here and in the report.
