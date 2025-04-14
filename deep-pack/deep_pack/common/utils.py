from typing import Union
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import torch as th
import os
import random
import numpy as np

def get_schedule_fn(initial_value: Union[float, str]) -> Schedule:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)
    # force to set the seed for generating random numbers on all GPUs
    th.cuda.manual_seed_all(seed)
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    th.use_deterministic_algorithms(True)
    
    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False