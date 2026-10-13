"""Tests that verify the results printed by tensor_manipulation.py.

Run from this folder with the virtual environment active:

    pytest -v test_tensor_manipulation.py
"""

import pytest
import torch


@pytest.fixture
def tens():
    """Integers 0, 10, ..., 90."""
    return torch.arange(0, 100, 10)


@pytest.fixture
def x():
    """Floats 1.0 to 7.0, built fresh for each test so edits do not leak."""
    return torch.arange(1.0, 8.0)


# --- Aggregation ------------------------------------------------------------


def test_min_max_sum(tens):
    assert tens.min().item() == 0
    assert tens.max().item() == 90
    assert tens.sum().item() == 450


def test_mean_fails_on_integer_tensor(tens):
    assert tens.dtype == torch.int64
    with pytest.raises(RuntimeError):
        tens.mean()


def test_mean_after_converting_to_float(tens):
    assert tens.type(torch.float32).mean().item() == 45.0


def test_functions_match_methods(tens):
    assert torch.max(tens) == tens.max()
    assert torch.min(tens) == tens.min()
    assert torch.sum(tens) == tens.sum()
    assert torch.mean(tens.type(torch.float32)) == tens.type(torch.float32).mean()


def test_argmax_and_argmin_return_positions():
    tensor = torch.arange(10, 100, 10)
    assert tensor.argmax().item() == 8
    assert tensor.argmin().item() == 0
    # The position points back at the value.
    assert tensor[tensor.argmax()].item() == 90
    assert tensor[tensor.argmin()].item() == 10


# --- Datatypes --------------------------------------------------------------


def test_float_arange_defaults_to_float32():
    assert torch.arange(10.0, 100.0, 10.0).dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_type_converts_and_keeps_values(dtype):
    tensor = torch.arange(10.0, 100.0, 10.0)
    converted = tensor.type(dtype)
    assert converted.dtype == dtype
    assert converted.tolist() == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # The original keeps its datatype.
    assert tensor.dtype == torch.float32


# --- Reshape and view -------------------------------------------------------


def test_reshape_changes_shape_not_values(x):
    x_reshaped = x.reshape(1, 7)
    assert x_reshaped.shape == (1, 7)
    assert x_reshaped.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


def test_reshape_rejects_incompatible_shape(x):
    with pytest.raises(RuntimeError):
        x.reshape(2, 7)


def test_view_shares_data_with_original(x):
    z = x.view(1, 7)
    assert z.shape == (1, 7)
    z[:, 0] = 5
    assert x.tolist() == [5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


# --- Stack ------------------------------------------------------------------


def test_stack_on_dim_0_makes_rows(x):
    x_stacked = torch.stack([x, x, x, x], dim=0)
    assert x_stacked.shape == (4, 7)
    assert torch.equal(x_stacked[0], x)


def test_stack_on_dim_1_makes_columns(x):
    x_stacked = torch.stack([x, x, x, x], dim=1)
    assert x_stacked.shape == (7, 4)
    assert torch.equal(x_stacked[:, 0], x)


# --- Squeeze and unsqueeze --------------------------------------------------


def test_squeeze_removes_size_1_dimensions(x):
    x_reshaped = x.reshape(1, 7)
    x_squeezed = x_reshaped.squeeze()
    assert x_squeezed.shape == (7,)
    assert torch.equal(x_squeezed, x)


@pytest.mark.parametrize("dim, expected", [(0, (1, 7)), (1, (7, 1))])
def test_unsqueeze_adds_a_size_1_dimension(x, dim, expected):
    assert x.unsqueeze(dim=dim).shape == expected


# --- Permute ----------------------------------------------------------------


def test_permute_reorders_axes():
    x_original = torch.rand(size=(224, 224, 3))
    x_permuted = x_original.permute(2, 0, 1)
    assert x_permuted.shape == (3, 224, 224)
    # The value at (height 5, width 7, channel 1) moved to (channel 1, height 5, width 7).
    assert x_permuted[1, 5, 7] == x_original[5, 7, 1]


def test_permute_shares_data_with_original():
    x_original = torch.rand(size=(224, 224, 3))
    x_permuted = x_original.permute(2, 0, 1)
    x_original[0, 0, 0] = 123.0
    assert x_permuted[0, 0, 0].item() == 123.0
