import os
import random

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
