import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.deterministic = True
    

