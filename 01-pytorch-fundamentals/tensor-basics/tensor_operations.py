"""Tensor operations: arithmetic, element-wise vs matrix multiplication, and timing.

Run with the virtual environment active:

    python tensor_operations.py
"""

import time

import torch


def dot_by_hand(tensor):
    """Multiply a 1-D tensor by itself and sum the results, using a Python loop.

    This gives the same answer as torch.matmul(tensor, tensor). It exists only
    to show how much slower a loop is; use torch.matmul in real code.
    """
    value = 0
    for i in range(len(tensor)):
        value += tensor[i] * tensor[i]
    return value


def basic_operations():
    print("=== Basic operations ===")
    tensor = torch.tensor([1, 2, 3])
    print(f"tensor:               {tensor}")

    # Arithmetic with a number applies to every element.
    print(f"tensor + 10:          {tensor + 10}")
    print(f"tensor * 10:          {tensor * 10}")

    # Neither line above changed the tensor: operations return a new tensor.
    print(f"tensor (unchanged):   {tensor}")

    # To keep a result, assign it back to the variable.
    tensor = tensor - 10
    print(f"after tensor - 10:    {tensor}")
    tensor = tensor + 10
    print(f"after tensor + 10:    {tensor}")

    # The same operations exist as functions. The operators are more common.
    print(f"torch.multiply(t, 10): {torch.multiply(tensor, 10)}")
    print(f"torch.add(t, 10):      {torch.add(tensor, 10)}")
    print(f"tensor (unchanged):    {tensor}")

    # Tensor * tensor is element-wise: index 0 with 0, 1 with 1, 2 with 2.
    print(f"{tensor} * {tensor} = {tensor * tensor}")
    print()


def matrix_multiplication():
    print("=== Matrix multiplication ===")
    tensor = torch.tensor([1, 2, 3])
    print(f"tensor: {tensor}, shape: {tensor.shape}")

    # Element-wise multiplies matching positions:      [1*1, 2*2, 3*3] = [1, 4, 9]
    # Matrix multiplication also adds them together:    1*1 + 2*2 + 3*3 = 14
    print(f"element-wise, tensor * tensor:       {tensor * tensor}")
    print(f"matrix, torch.matmul(tensor, tensor): {torch.matmul(tensor, tensor)!r}")
    print(f"matrix, tensor @ tensor:              {(tensor @ tensor)!r}")
    print()

    # Rule 1: the inner dimensions must match.
    # Rule 2: the result has the shape of the outer dimensions.
    a = torch.rand(2, 3)
    b = torch.rand(3, 2)
    print("=== Shape rules ===")
    print(f"(2, 3) @ (3, 2) -> {tuple((a @ b).shape)}")
    print(f"(3, 2) @ (2, 3) -> {tuple((b @ a).shape)}")
    try:
        b @ b
    except RuntimeError as error:
        print(f"(3, 2) @ (3, 2) -> fails: {error}")
    print()


def timing_comparison():
    print("=== Loop vs torch.matmul ===")
    # A larger tensor than [1, 2, 3], so the difference is measurable.
    tensor = torch.rand(10_000)

    start = time.perf_counter()
    by_hand = dot_by_hand(tensor)
    loop_seconds = time.perf_counter() - start

    start = time.perf_counter()
    built_in = torch.matmul(tensor, tensor)
    matmul_seconds = time.perf_counter() - start

    print(f"loop:         {by_hand:.4f} in {loop_seconds * 1000:.3f} ms")
    print(f"torch.matmul: {built_in:.4f} in {matmul_seconds * 1000:.3f} ms")
    print(f"torch.matmul was about {loop_seconds / matmul_seconds:,.0f}x faster")


def main():
    basic_operations()
    matrix_multiplication()
    timing_comparison()


if __name__ == "__main__":
    main()
