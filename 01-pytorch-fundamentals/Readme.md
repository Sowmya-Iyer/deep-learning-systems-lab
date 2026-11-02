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

## Weeks 1–2 notes

| Topic | Code |
| ----- | ---- |
| [Tensors](#tensors) | [tensor_operations.py](tensor-basics/tensor_operations.py), [tensor_manipulation.py](tensor-basics/tensor_manipulation.py) |
| [Autograd](#autograd) | [core_concepts.py](nn-basics/core_concepts.py), [autograd_check.py](nn-basics/autograd_check.py) |
| [Linear regression](#linear-regression) | [linear_regression.py](nn-basics/linear_regression.py) |
| [Data loading](#data-loading) | [datasets.py](data_utils/datasets.py) |
| [MLP](#mlp) | [mlp.py](mlp/mlp.py) |
| [Small CNN](#small-cnn) | [smallcnn.py](CNN/smallcnn.py) |
| [Optimizer comparison](#optimizer-comparison) | [cnn_optimizer_comparison.py](CNN/cnn_optimizer_comparison.py) |

### Tensors

[tensor_operations.py](tensor-basics/tensor_operations.py) covers arithmetic, the difference between `*` and `@`, the matrix shape rules, and how much slower a Python loop is than `torch.matmul`. [tensor_manipulation.py](tensor-basics/tensor_manipulation.py) has the manipulation exercises.

- `*` multiplies matching elements. `@` also sums them, so `[1, 2, 3] @ [1, 2, 3]` is 14.
- For `@`, the inner dimensions must match, and the result takes the outer dimensions.
- Operations return a new tensor unless the result is assigned back.

Tests: [create](tensor-basics/test_tensor_create.py), [factory](tensor-basics/test_tensor_factory.py), [attributes](tensor-basics/test_tensor_attributes.py), [indexing](tensor-basics/test_tensor_indexing.py), [operations](tensor-basics/test_tensor_operations.py), [manipulation](tensor-basics/test_tensor_manipulation.py).

### Autograd

[core_concepts.py](nn-basics/core_concepts.py) covers broadcasting, strides and the chain rule. For `L = (10x + 5)²` at `x = 5`, `dL/dx = 20(10x + 5) = 1100`.

[autograd_check.py](nn-basics/autograd_check.py) computes the gradient of `x²` at `x = 3` three ways: by hand, with autograd, and numerically with a central difference. All three must agree. It uses `float64` so the numerical estimate is precise enough to compare.

### Linear regression

[linear_regression.py](nn-basics/linear_regression.py) fits `y = wx + b` without `nn.Module` or an optimizer, which shows each step of the training loop: predict, compute the loss, call `backward()`, update inside `torch.no_grad()`, then clear the gradients. Gradients add up across calls to `backward()`, so skipping that last step would corrupt the next update.

### Data loading

[datasets.py](data_utils/datasets.py) loads FashionMNIST with batch size 64 and normalizes pixels to about -1 to 1. Labels are one-hot, so accuracy compares `y.argmax(1)` with the predicted class. The transforms move each sample to the GPU one at a time, which works here but would not suit worker processes.

### MLP

[mlp.py](mlp/mlp.py) flattens each image and passes it through two hidden layers of 512 units. With plain SGD at a learning rate of 0.001, it reached **84.5%** test accuracy and a loss of 0.4366.

The model outputs logits, and `nn.CrossEntropyLoss` applies the softmax itself, so the model has no softmax layer.

### Small CNN

[smallcnn.py](CNN/smallcnn.py) has two convolution and pooling blocks, then two linear layers. Pooling reduces each image from 28×28 to 14×14 to 7×7. With Adam at a learning rate of 0.001, it reached **89.79%** test accuracy and a loss of 0.2793 after 5 epochs. Test loss was still falling at the end, so 5 epochs is probably too few to show its best result.

### Optimizer comparison

[cnn_optimizer_comparison.py](CNN/cnn_optimizer_comparison.py) trains the small CNN for 5 epochs with every combination of optimizer, learning rate and seed (42, 123, 2026). SGD uses momentum 0.9, and both optimizers use weight decay 1e-4. Each run resets the seed, so runs with the same seed start from the same weights and shuffle order.

Final results, averaged over the 3 seeds:

| Optimizer | Learning rate | Mean test accuracy | Lowest to highest | Mean test loss |
| --------- | ------------: | -----------------: | ----------------- | --------------: |
| AdamW | 0.001 | 89.48% | 89.00–89.79 | 0.2912 |
| AdamW | 0.01 | 89.10% | 88.85–89.37 | 0.3171 |
| SGD | 0.01 | 88.94% | 88.62–89.22 | 0.3041 |
| AdamW | 0.0001 | 84.96% | 84.57–85.17 | 0.4220 |
| SGD | 0.001 | 82.26% | 80.94–83.39 | 0.4924 |
| SGD | 0.0001 | 68.45% | 67.15–69.74 | 0.8310 |

- **The top three are effectively tied on accuracy.** They are 0.54 points apart, and one setup varies by up to 0.79 points between seeds.
- **SGD depends more on the learning rate.** Going from 0.01 to 0.0001 costs SGD 20.5 points and AdamW 4.1.
- **SGD at 0.0001 starts slowly.** Its first-epoch train loss is about 2.28, close to the 2.303 (ln 10) expected from random guessing across 10 classes.
- **AdamW at 0.01 is probably too high.** Its final train loss matches AdamW at 0.001, but its test loss is worse.

AdamW at 0.001 is the default for the remaining exercises.

**Limits:**

- Five epochs measures how quickly each setup learns, not its final quality.
- These are test-set results, so they cannot be used to select a final setting. That choice needs a validation split.
- MLP and CNN results above are not a fair architecture comparison, because their optimizers and epochs differ.

### Still to do

- [ ] Run the MLP through the same comparison
- [ ] Overfit a tiny batch of about 32 images
- [ ] Show what `detach()` changes
- [ ] Compare initialization and normalization choices
- [ ] Add a validation split
- [ ] Write up a training failure, starting with the `argmax` error in [mlp.py](mlp/mlp.py)
- [ ] Fix [core_concepts.py](nn-basics/core_concepts.py) so the chain rule example prints `new_x.grad`, not `x.grad`
