import torch
import pytest

GRID = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

@pytest.fixture
def grid():
    return torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# --- Indexing and slicing ---------------------------------------------------


def test_first_row(grid):
    assert grid[0].tolist() == [1, 2, 3, 4]


def test_first_column(grid):
    assert grid[:, 0].tolist() == [1, 5, 9, 13]


def test_last_column(grid):
    assert grid[..., -1].tolist() == [4, 8, 12, 16]


def test_slice_assignment(grid):
    grid[:, 1] = 0
    grid[2, :] = 0
    expected = torch.tensor(
        [
            [1, 0, 3, 4],
            [5, 0, 7, 8],
            [0, 0, 0, 0],
            [13, 0, 15, 16],
        ]
    )
    assert torch.equal(grid, expected)
