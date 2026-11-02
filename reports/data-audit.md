# Data audit

Week 3, bounded to a few hours. Confirm these facts from each dataset's own page before writing model code. The notes are starting points, not verified facts.

| Dataset | Used for | Check first | Notes so far | License | Size | Decision |
| ------- | -------- | ----------- | ------------ | ------- | ---: | -------- |
| [LOCO](https://github.com/tum-fml/loco) | Projects 1, 2 | Class counts, split unit, unannotated images | Forklift annotations are few, so audit before choosing classes | — | — | — |
| [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) | Project 3, first choice | Is download access open? Which traverses have [RTK ground truth](https://arxiv.org/abs/2002.10152)? What odometry can the EKF use? Which few traverses cover different conditions? | Registration required. RTK release terms still to confirm | CC BY-NC-SA 4.0 per the site | — | — |
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Project 3, fallback | What are the license and official download sizes? Do the raw recordings supply usable IMU data? | Poses are published for sequences 00–10. Less change in weather and light, so it answers the condition question less well | — | — | — |
| [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) | Projects 1, 2, fallback | What do the official terms say? Sources disagree | Use only if LOCO fails the rule below | — | — | — |
| [nuImages](https://www.nuscenes.org/nuimages) | Projects 1, 2, fallback | What are the license and download size? | Use only if LOCO fails the rule below | — | — | — |
| [SimReady Warehouse](https://hyper.ai/en/datasets/nvidia-physical-ai-simready-warehouse-openusd) | Project 4 | Can it render readable labels? | Synthetic 3D assets | — | — | — |

## Decisions

- **Project 1 and 2 classes and split unit:** —
- **LOCO, keep or replace:** —
- **Project 3 dataset:** —
- **Project 3 reference-pose description:** —
- **Odometry source:** real or simulated, with reason —
- **Project 4 synthetic label source:** —
- **Project 4 real-image holdout source:** —

**LOCO rule.** Keep LOCO if at least three useful classes each have 500 training crops and 100 test crops after splitting. That leaves 50 examples per class at the 10% label budget. Otherwise, replace it with BDD100K or nuImages for Projects 1 and 2, and rename those projects to match. Decide by Nov 8.

Download only a few RobotCar traverses, and record their total size here.

If no dataset has usable odometry, Project 3 simulates it from reference poses. Record the noise model here and in the report.
