import numpy as np
import pytest


def test_torch_and_numpy_interoperate():
    torch = pytest.importorskip("torch")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(a)
    assert t.sum().item() == 15.0
    assert np.array_equal(t.numpy(), a)
