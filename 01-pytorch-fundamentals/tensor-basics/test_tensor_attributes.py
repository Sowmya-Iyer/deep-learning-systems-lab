import torch
import pytest

#--- Attributes and devices -------------------------------------------------
 
 
def test_tensor_attributes():
    tensor = torch.rand(3, 4)
    assert tensor.shape == torch.Size([3, 4])
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
 
 
def test_move_to_accelerator_keeps_values():
    if not torch.accelerator.is_available():
        pytest.skip("no accelerator available on this machine")
    device = torch.accelerator.current_accelerator()
    original = torch.rand(3, 4)
    moved = original.to(device)
    assert moved.device.type == device.type
    assert torch.equal(moved.cpu(), original)
 
 
def test_tensor_attributes():
    tensor = torch.rand(3, 4)
    assert tensor.shape == torch.Size([3, 4])
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
 
 
def test_move_to_accelerator_keeps_values():
    if not torch.accelerator.is_available():
        pytest.skip("no accelerator available on this machine")
    device = torch.accelerator.current_accelerator()
    original = torch.rand(3, 4)
    moved = original.to(device)
    assert moved.device.type == device.type
    assert torch.equal(moved.cpu(), original)
