import random

import numpy as np
import pytest
import torch

from palu_tt import device as dev


@pytest.fixture(autouse=True)
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    torch.set_num_threads(12)


@pytest.fixture(scope="session")
def mesh_device():
    md = dev.open_n300()
    yield md
    dev.close(md)
