# 01-pytorch-fundamentals

PyTorch practice code, part of the `deep-learning-systems-lab` repository. Dependencies live in a local virtual environment (`.torch-venv`) that is not committed to Git, so every machine rebuilds it from `requirements.txt` using the steps below.

The environment uses the CUDA 13.2 build of PyTorch, which runs on an NVIDIA GPU. A CPU-only alternative is given in step 4.

## Prerequisites

- Git
- Python 3.10 to 3.14
- For GPU use: an NVIDIA GPU with a driver that supports CUDA 13.2 or newer

Check them from a terminal:

```
git --version
python --version
nvidia-smi
```

`nvidia-smi` prints a table with the GPU name and, in the top-right corner, the highest CUDA version the driver supports. That number must be 13.2 or higher. If the command is not found, install or update the NVIDIA driver, or use the CPU-only install in step 4.

No separate CUDA toolkit install is needed; the PyTorch package includes the CUDA libraries.
