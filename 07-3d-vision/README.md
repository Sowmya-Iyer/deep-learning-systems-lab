# 07-3d-vision

Geometry, scene representation, warehouse depth and occupancy, and deployment for [Project 5](../PROJECTS.md). Weeks 17-18 and 21-24.

## Layout

```
07-3d-vision/
├── geometry/       # camera models, Structure from Motion, bundle adjustment
├── splatting/      # 3D Gaussian Splatting from recovered poses
├── perception/     # depth and occupancy, with and without rendered views
├── deployment/     # export, quantization, latency measurements
└── configs/
```

## Compute

Start geometry and splatting on the laptop with one small scene. Train depth and occupancy on a rented GPU. Measure deployment latency on the target device, and name that device in the report.

## Done when

- [ ] Structure from Motion and bundle adjustment run on one warehouse sequence, with reprojection and pose error measured
- [ ] Gaussian Splatting is scored on held-out frames
- [ ] Depth and occupancy are trained with and without rendered views, and the gain is measured, even if it is zero
- [ ] One model is exported and quantized, with latency, accuracy and size reported
- [ ] One demo runs from warehouse images to a 3D scene and a perception output

Temporal tracking and a NeRF comparison are optional. Results go in the [Project 5 report](../reports/project-5-warehouse-3d/report.md).
