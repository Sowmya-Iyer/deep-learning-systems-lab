# CUDA Review Notes

## 1.1 Introduction

### 1.1.1 GPU

**CUDA = NVIDIA's platform/programming model for general-purpose GPU computing.**

---

## 1.1.2 Why GPUs?

### CPU vs GPU

| CPU | GPU |
|---|---|
| Optimized for **latency** | Optimized for **throughput** |
| Few powerful cores | Many parallel execution units |
| Few concurrent threads | Thousands of concurrent threads |
| More cache/control logic | More resources for computation |
| Strong serial performance | Strong parallel performance |

### Core Idea

> **CPU:** Make a small number of threads very fast.  
> **GPU:** Execute a huge number of threads concurrently.

