# 04-distributed

DDP, memory analysis and optional FSDP for [Project 1](../PROJECTS.md). Weeks 19-20, on a rented machine with two GPUs.

## Layout

```
04-distributed/
├── ddp/            # DDP training entry point, sampler and checkpoint checks
├── memory/         # activation checkpointing measurements
├── fsdp/           # conditional: only if the memory results justify it
└── configs/
```

## Done when

- [ ] DDP training runs on the shared `dlsl` pipeline
- [ ] `DistributedSampler` behavior is verified: shards are disjoint and reshuffle each epoch
- [ ] A checkpoint saved under DDP resumes correctly
- [ ] Throughput, time-to-target quality, scaling efficiency and cost are measured
- [ ] Activation checkpointing is measured for memory saved against speed lost

## Rule

Multi-GPU claims require measured GPU runs. Debug on the laptop with `gloo` or a single process before renting, then rent one machine for the whole measurement so every row shares the same hardware. Use four GPUs only if the two-GPU result justifies it.

| GPUs | Global batch | Samples/second | Time to target | Peak memory/GPU | Scaling efficiency | Cost |
| ---: | -----------: | -------------: | -------------: | --------------: | -----------------: | ---: |
|    1 |            — |              — |              — |               — |               100% |    — |
|    2 |            — |              — |              — |               — |                  — |    — |

Results go in the [Project 1 report](../reports/project-1-vit-scaling/report.md).
