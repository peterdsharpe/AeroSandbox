from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
from aerosandbox.performance import OperatingPoint
import os
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airfoil
from typing import Union, List, Dict
import tempfile
import warnings


class XFoil(ExplicitAnalysis):
    """

    An interface to XFoil, a 2D airfoil analysis tool developed by Mark Drela at MIT.

    Requires XFoil to be on your computer; XFoil is available here: https://web.mit.edu/drela/Public/web/xfoil/

    It is recommended (but not required) that you add XFoil to your system PATH environment variable such that it can
    be called with the command `xfoil`. If this is not the case, you need to specify the path to your XFoil
    executable using the `xfoil_command` argument of the constructor.

    Usage example:

    >>> xf = XFoil(
    >>>     airfoil=Airfoil("naca2412").repanel(n_points_per_side=100),
    >>>     Re=1e6,
    >>> )
    >>>
    >>> result_at_single_alpha = xf.alpha(5)
    >>> result_at_several_CLs = xf.cl([0.5, 0.7, 0.8, 0.9])
    >>> result_at_multiple_alphas = xf.alpha([3, 5, 60]) # Note: if a result does not converge (such as the 60 degree case here), it will not be included in the results.

    """

    def __init__(self,
                 airfoil: Airfoil,
                 Re: float = 0.,
                 mach: float = 0.,
                 n_crit: float = 9.,
                 xtr_upper: float = 1.,
                 xtr_lower: float = 1.,
                 max_iter: int = 100,
                 xfoil_command: str = "xfoil",
                 verbose: bool = False,
                 working_directory: str = None,
                 ):
        """
        Interface to XFoil.

        Args:

            airfoil: The angle of attack [degrees]

            Re: The chord-referenced Reynolds number

            mach: The freestream Mach number

            n_crit: The critical Tollmein-Schlichting wave amplification factor

            xtr_upper: The upper-surface trip location [x/c]

            xtr_lower: The lower-surface trip location [x/c]

            max_iter: How many iterations should we let XFoil do?

            xfoil_command: The command-line argument to call XFoil.

                * If XFoil is on your system PATH, then you can just leave this as "xfoil".

                * If XFoil is not on your system PATH, then you should provide a filepath to the XFoil executable.

                Note that XFoil is not on your PATH by default. To tell if XFoil is not on your system PATH,
                open up a terminal and type "xfoil".

                    * If the XFoil menu appears, it's on your PATH.

                    * If you get something like "'xfoil' is not recognized as an internal or external command..." or
                    "Command 'xfoil' not found, did you mean...", then it is not on your PATH and you'll need to
                    specify the location of your XFoil executable as a string.

                To add XFoil to your path, modify your system's environmental variables. (Google how to do this for
                your OS.)

        """
        self.airfoil = airfoil
        self.Re = Re
        self.mach = mach
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.max_iter = max_iter
        self.xfoil_command = xfoil_command
        self.verbose = verbose
        self.working_directory = working_directory

    def _default_run_file_contents(self) -> List[str]:
        run_file_contents = []

        # Disable graphics
        run_file_contents += [
            "plop",
            "g",
            "",
        ]

        # Enter oper mode
        run_file_contents += [
            "oper",
        ]

        # Handle Re
        if self.Re != 0:
            run_file_contents += [
                f"re {self.Re}",
                "v",
            ]

        # Handle mach
        run_file_contents += [
            f"m {self.mach}",
        ]

        # Handle iterations
        run_file_contents += [
            f"iter {self.max_iter}",
        ]

        # Handle trips and ncrit
        run_file_contents += [
            "vpar",
            f"xtr {self.xtr_upper} {self.xtr_lower}",
            f"n {self.n_crit}",
            "",
        ]

        # Set polar accumulation
        run_file_contents += [
            "pacc",
            "",
            "",
        ]

        return run_file_contents

    def _run_xfoil(self,
                   run_command: str,
                   ) -> Dict[str, np.ndarray]:
        """
        Private function to run XFoil.

        Args: run_command: A string with any XFoil inputs that you'd like. By default, you start off within the OPER
        menu. All of the inputs indicated in the constructor have been set already, but you can override them here (for
        this run only) if you want.

        Returns: A dictionary containing all converged solutions obtained with your inputs.

        """
        # Set up a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory) # For debugging

            # Designate an intermediate file for file I/O
            output_filename = "output.txt"

            # Handle the airfoil file
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            # Handle the run file
            run_file_contents = self._default_run_file_contents()
            run_file_contents += [run_command]
            run_file_contents += [
                "pwrt",
                f"{output_filename}",
                "y",
                "",
                "quit"
            ]
            run_file = "run_file.txt"
            with open(directory / run_file, "w+") as f:
                f.write(
                    "\n".join(run_file_contents)
                )

            ### Set up the run command
            command = f'{self.xfoil_command} {airfoil_file} < {run_file}'

            ### Execute
            subprocess.call(
                command,
                shell=True,
                cwd=directory,
                stdout=None if self.verbose else subprocess.DEVNULL
            )

            ### Parse the polar
            columns = [
                "alpha",
                "CL",
                "CD",
                "CDp",
                "CM",
                "xtr_upper",
                "xtr_lower"
            ]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output_data = np.genfromtxt(
                    directory / output_filename,
                    skip_header=12,
                    usecols=np.arange(len(columns))
                ).reshape(-1, len(columns))

            has_valid_inputs = len(output_data) != 0

            return {
                k: output_data[:, index] if has_valid_inputs else np.array([])
                for index, k in enumerate(columns)
            }

    def alpha(self,
              alpha: Union[float, np.ndarray]
              ) -> Dict[str, np.ndarray]:
        """
        Execute XFoil at a given angle of attack, or at a sequence of angles of attack.

        Args:
            alpha: The angle of attack [degrees]. Can be either a float or an iterable of floats, such as an array.

        Returns: A dictionary with the XFoil results. Dictionary values are arrays; they may not be the same shape as
        your input array if some points did not converge.

        """
        alphas = np.array(alpha).reshape(-1)

        return self._run_xfoil(
            "\n".join([
                f"a {a}"
                for a in alphas
            ])
        )

    def cl(self,
           cl: Union[float, np.ndarray]
           ) -> Dict[str, np.ndarray]:
        """
        Execute XFoil at a given lift coefficient, or at a sequence of lift coefficients.

        Args:
            cl: The lift coefficient [-]. Can be either a float or an iterable of floats, such as an array.

        Returns: A dictionary with the XFoil results. Dictionary values are arrays; they may not be the same shape as
        your input array if some points did not converge.

        """
        cls = np.array(cl).reshape(-1)

        return self._run_xfoil(
            "\n".join([
                f"cl {c}"
                for c in cls
            ])
        )


if __name__ == '__main__':
    xf = XFoil(
        airfoil=Airfoil("naca2412").repanel(n_points_per_side=100),
        Re=1e6,
    )
    result_at_single_alpha = xf.alpha(5)
    result_at_several_CLs = xf.cl([0.5, 0.7, 0.8, 0.9])
    result_at_multiple_alphas = xf.alpha([3, 5, 60])  # Note: if a result does
