import torch
import pytest

@pytest.mark.parametrize("factory", [torch.rand, torch.ones, torch.zeros])

def test_factory_shape_and_dtype(factory):
    shape = (2, 3)
    tensor = factory(shape)
    assert tensor.shape == shape
    assert tensor.dtype == torch.float32
 
 
def test_ones_values():
    assert torch.equal(torch.ones((2, 3)), torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
 
 
def test_zeros_values():
    assert torch.equal(torch.zeros((2, 3)), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
 
 
def test_rand_values_are_in_unit_interval():
    rand_tensor = torch.rand((2, 3))
    assert ((rand_tensor >= 0) & (rand_tensor < 1)).all()
 
 
def test_rand_is_reproducible_with_a_seed():
    torch.manual_seed(0)
    first = torch.rand((2, 3))
    torch.manual_seed(0)
    second = torch.rand((2, 3))
    assert torch.equal(first, second)
 
