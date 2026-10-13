"""Tensor aggregation, datatypes, and reshaping.

Run with the virtual environment active:

    python tensor_manipulation.py
"""

import torch


def aggregation():
    print("=== Aggregation: min, max, mean, sum ===")
    x = torch.arange(0, 100, 10)
    print(f"x: {x}")

    print(f"Minimum: {x.min()}")
    print(f"Maximum: {x.max()}")
    print(f"Sum:     {x.sum()}")

    # mean() needs a floating-point tensor. x holds integers (int64), so
    # calling it directly raises an error.
    try:
        x.mean()
    except RuntimeError as error:
        print(f"x.mean() on integers fails: {error}")

    # Convert to float32 first, then take the mean.
    print(f"Mean:    {x.type(torch.float32).mean()}")

    # The same four operations as torch functions instead of methods.
    print(
        "As functions: "
        f"{torch.max(x)!r}, {torch.min(x)!r}, "
        f"{torch.mean(x.type(torch.float32))!r}, {torch.sum(x)!r}"
    )
    print()

    print("=== Positional min/max ===")
    tensor = torch.arange(10, 100, 10)
    print(f"Tensor: {tensor}")

    # argmax/argmin return the index of the value, not the value itself.
    print(f"Index where max value occurs: {tensor.argmax()}")
    print(f"Index where min value occurs: {tensor.argmin()}")
    print()


def datatypes():
    print("=== Changing datatype ===")
    # Writing the numbers with a decimal point makes a float tensor.
    # The default float type is float32.
    tensor = torch.arange(10.0, 100.0, 10.0)
    print(f"tensor:         {tensor}")
    print(f"tensor.dtype:   {tensor.dtype}")

    # .type() returns a new tensor in the requested datatype.
    tensor_float16 = tensor.type(torch.float16)
    print(f"tensor_float16: {tensor_float16}")

    tensor_int8 = tensor.type(torch.int8)
    print(f"tensor_int8:    {tensor_int8}")

    # The original is unchanged.
    print(f"tensor.dtype:   {tensor.dtype}")
    print()


def reshaping():
    print("=== Reshape ===")
    x = torch.arange(1.0, 8.0)
    print(f"x: {x}, shape: {x.shape}")

    # Same seven values, now arranged as 1 row of 7 columns.
    x_reshaped = x.reshape(1, 7)
    print(f"x_reshaped: {x_reshaped}, shape: {x_reshaped.shape}")
    print()

    print("=== View ===")
    # A view has a different shape but shares its data with the original.
    z = x.view(1, 7)
    print(f"z: {z}, shape: {z.shape}")

    # Because the data is shared, changing z also changes x.
    z[:, 0] = 5
    print("after z[:, 0] = 5")
    print(f"z: {z}")
    print(f"x: {x}")

    # reshape() also returns a view when it can, so x_reshaped changed too.
    print(f"x_reshaped: {x_reshaped}")
    print()

    print("=== Stack ===")
    # dim=0 stacks the tensors as rows: 4 tensors of 7 values -> shape (4, 7).
    x_stacked = torch.stack([x, x, x, x], dim=0)
    print(f"stacked on dim=0, shape {tuple(x_stacked.shape)}:\n{x_stacked}")

    # dim=1 stacks them as columns instead -> shape (7, 4).
    x_stacked_columns = torch.stack([x, x, x, x], dim=1)
    print(f"stacked on dim=1, shape {tuple(x_stacked_columns.shape)}:\n{x_stacked_columns}")
    print()

    print("=== Squeeze ===")
    # squeeze() removes every dimension of size 1.
    print(f"Previous tensor: {x_reshaped}")
    print(f"Previous shape:  {x_reshaped.shape}")
    x_squeezed = x_reshaped.squeeze()
    print(f"New tensor:      {x_squeezed}")
    print(f"New shape:       {x_squeezed.shape}")
    print()

    print("=== Unsqueeze ===")
    # unsqueeze() does the reverse: it adds a dimension of size 1 at `dim`.
    print(f"Previous tensor: {x_squeezed}")
    print(f"Previous shape:  {x_squeezed.shape}")
    x_unsqueezed = x_squeezed.unsqueeze(dim=0)
    print(f"New tensor:      {x_unsqueezed}")
    print(f"New shape:       {x_unsqueezed.shape}")
    print()

    print("=== Permute ===")
    # An image stored as (height, width, colour channels).
    x_original = torch.rand(size=(224, 224, 3))

    # Reorder the axes to (colour channels, height, width).
    # permute(2, 0, 1) means: new axis 0 is old axis 2, new 1 is old 0, new 2 is old 1.
    x_permuted = x_original.permute(2, 0, 1)
    print(f"Previous shape: {x_original.shape}")
    print(f"New shape:      {x_permuted.shape}")

    # permute() returns a view, so a change to one shows up in the other.
    x_original[0, 0, 0] = 123.0
    print(f"x_original[0, 0, 0] set to 123 -> x_permuted[0, 0, 0] is {x_permuted[0, 0, 0]}")


def main():
    aggregation()
    datatypes()
    reshaping()


if __name__ == "__main__":
    main()
