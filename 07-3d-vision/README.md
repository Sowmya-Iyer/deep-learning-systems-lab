# 07-3d-vision

Optional extension. Start only with spare project time; weeks 21-24 stay reserved for interview preparation. See [PROJECTS.md](../PROJECTS.md).

**Question.** How does pose quality affect novel-view quality on one warehouse sequence?

## Layout

```
07-3d-vision/
├── geometry/       # camera models, Structure from Motion, bundle adjustment
├── splatting/      # 3D Gaussian Splatting
└── configs/
```

## Done when

- [ ] Structure from Motion and bundle adjustment run on one sequence with enough camera overlap
- [ ] Reprojection error and pose error are measured
- [ ] Gaussian Splatting is scored on held-out views for each pose source

Good novel views do not prove accurate metric geometry or useful data augmentation, so make neither claim. Record results in this README.
