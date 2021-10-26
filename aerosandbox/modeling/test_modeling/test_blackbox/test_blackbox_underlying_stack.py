import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List

opti = asb.Opti()

x = opti.variable(init_guess=2)

import casadi as cas


class MyGamma(cas.Callback):
    def __init__(self):
        super().__init__()
        self.construct(
            "Gamma",
            {
                "enable_fd": True
            }
        )

    def eval(self,
             args: List[cas.DM]
             ) -> List[cas.DM]:
        from math import gamma

        return [
            gamma(args[0])
        ]


mygamma = MyGamma()

opti.minimize(mygamma(x))
opti.subject_to(mygamma(np.sin(x)) > 0.1)

sol = opti.solve()
