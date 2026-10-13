"""Tests that verify the results printed by tensor_operations.py.

Run from this folder with the virtual environment active:

    pytest -v test_tensor_operations.py
"""

import pytest
import torch

from tensor_operations import dot_by_hand


@pytest.fixture
def tensor():
    return torch.tensor([1, 2, 3])


# --- Basic operations -------------------------------------------------------


def test_add_number(tensor):
    assert (tensor + 10).tolist() == [11, 12, 13]


def test_multiply_by_number(tensor):
    assert (tensor * 10).tolist() == [10, 20, 30]


def test_operations_do_not_change_the_original(tensor):
    tensor + 10
    tensor * 10
    torch.multiply(tensor, 10)
    assert tensor.tolist() == [1, 2, 3]


def test_subtract_then_add_with_reassignment(tensor):
    tensor = tensor - 10
    assert tensor.tolist() == [-9, -8, -7]
    tensor = tensor + 10
    assert tensor.tolist() == [1, 2, 3]


def test_functions_match_operators(tensor):
    assert torch.equal(torch.multiply(tensor, 10), tensor * 10)
    assert torch.equal(torch.add(tensor, 10), tensor + 10)


def test_elementwise_multiplication(tensor):
    assert (tensor * tensor).tolist() == [1, 4, 9]


# --- Matrix multiplication --------------------------------------------------


def test_matmul_of_vector_with_itself(tensor):
    result = torch.matmul(tensor, tensor)
    assert result.ndim == 0
    assert result.item() == 14


def test_at_operator_matches_matmul(tensor):
    assert torch.equal(tensor @ tensor, torch.matmul(tensor, tensor))


def test_loop_matches_matmul(tensor):
    assert dot_by_hand(tensor).item() == torch.matmul(tensor, tensor).item()


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ((2, 3), (3, 2), (2, 2)),
        ((3, 2), (2, 3), (3, 3)),
    ],
)
def test_matmul_result_has_outer_dimensions(left, right, expected):
    result = torch.rand(left) @ torch.rand(right)
    assert result.shape == expected


def test_matmul_fails_when_inner_dimensions_differ():
    with pytest.raises(RuntimeError):
        torch.rand(3, 2) @ torch.rand(3, 2)
