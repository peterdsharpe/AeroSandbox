from aerosandbox import ImplicitAnalysis, Opti
import aerosandbox.numpy as np


class IBL2(ImplicitAnalysis):
    """
    An implicit analysis for a 2-dimensional integral boundary layer model.

    Implements a 2-equation dissipation-based model, partly based on:

        * Drela, Mark. "Aerodynamics of Viscous Fluids" textbook. Currently unpublished at the
        time of writing; contact Mark Drela (drela@mit.edu) to request a copy of the draft. References in the code
        here to "AVF Eq. XX" refer to equations in this book.




    """

    @ImplicitAnalysis.initialize
    def __init__(self,
                 streamwise_coordinate: np.ndarray,
                 edge_velocity: np.ndarray,
                 viscosity: float,
                 theta_0: float,
                 H_0: float = 2.6,
                 opti: Opti = None,
                 ):
        pass
