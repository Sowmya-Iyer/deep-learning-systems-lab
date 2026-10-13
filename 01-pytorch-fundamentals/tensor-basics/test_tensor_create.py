import numpy as np
import torch
import pytest

SCALAR = 5
VECTOR = [1, 2, 3, 4]
DATA = [[1, 2], [3, 4]]

@pytest.fixture
def x_data():
    """The 2x2 integer tensor built directly from DATA."""
    return torch.tensor(DATA)


@pytest.fixture
def x_scalar():
    return torch.tensor(SCALAR)

@pytest.fixture
def x_vector():
    return torch.tensor(VECTOR)


# --- Creating tensors -------------------------------------------------------

def test_tensor_from_scalar(x_scalar):
    assert x_scalar.ndim == 0
    assert x_scalar.item() == SCALAR

def test_tensor_from_vector(x_vector):
    assert x_vector.ndim == 1
    assert x_vector.shape == (4,)
    assert x_vector.tolist() == VECTOR
    
def test_tensor_from_data(x_data):
    assert x_data.shape == (2, 2)
    assert x_data.dtype == torch.int64
    assert x_data.tolist() == DATA


def test_tensor_from_numpy_has_same_values():
    np_array = np.array(DATA)
    x_np = torch.from_numpy(np_array)
    assert x_np.shape == (2, 2)
    assert x_np.tolist() == DATA


def test_tensor_from_numpy_shares_memory():
    np_array = np.array(DATA)
    x_np = torch.from_numpy(np_array)
    np_array[0, 0] = 99
    assert x_np[0, 0].item() == 99


def test_ones_like_keeps_shape_and_dtype(x_data):
    x_ones = torch.ones_like(x_data)
    assert x_ones.shape == x_data.shape
    assert x_ones.dtype == x_data.dtype
    assert torch.equal(x_ones, torch.tensor([[1, 1], [1, 1]]))


def test_rand_like_overrides_dtype(x_data):
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    assert x_rand.shape == x_data.shape
    assert x_rand.dtype == torch.float32
    # Random values cannot be compared exactly, so check their range instead.
    assert ((x_rand >= 0) & (x_rand < 1)).all()

def test_arange():
    zero_to_ten = torch.arange(start=0, end=10, step=1)
    assert torch.equal(zero_to_ten, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
