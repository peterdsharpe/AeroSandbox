import aerosandbox.numpy as np
from typing import List


def softmax_scalefree(x: List[float]) -> float:
    if len(x) == 1:
        return x[0]
    else:
        softness = np.max(np.array(
            [1e-6] + x
        )) * 0.01

        return np.softmax(
            *x,
            softness=softness
        )
