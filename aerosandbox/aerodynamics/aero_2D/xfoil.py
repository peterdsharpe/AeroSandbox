from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
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
                 xfoil_repanel: bool = True,
                 verbose: bool = False,
                 timeout: Union[float, int, None] = 30,
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

                To add XFoil to your path, modify your system's environment variables. (Google how to do this for
                your OS.)

            xfoil_repanel: Controls whether to allow XFoil to repanel your airfoil using its internal methods (PANE
            -> PPAR, both with default settings, 160 nodes)

            verbose: Controls whether or not XFoil output is printed to command line.

            timeout: Controls how long any individual XFoil run (i.e. alpha sweep) is allowed to run before the
            process is killed. Given in units of seconds. To disable timeout, set this to None.

            working_directory: Controls which working directory is used for the XFoil input and output files. By
            default, this is set to a TemporaryDirectory that is deleted after the run. However, you can set it to
            somewhere local for debugging purposes.

        """
        self.airfoil = airfoil
        self.Re = Re
        self.mach = mach
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.max_iter = max_iter
        self.xfoil_command = xfoil_command
        self.xfoil_repanel = xfoil_repanel
        self.verbose = verbose
        self.timeout = timeout
        self.working_directory = working_directory

        if np.length(self.airfoil.coordinates) > 401: # If the airfoil coordinates exceed Fortran array allocation
            self.xfoil_repanel = True


    def _default_keystrokes(self) -> List[str]:
        run_file_contents = []

        # Disable graphics
        run_file_contents += [
            "plop",
            "g",
            "",
        ]

        if self.xfoil_repanel:
            run_file_contents += [
                "pane",
                "ppar",
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

        Args: run_command: A string with any XFoil keystroke inputs that you'd like. By default, you start off within the OPER
        menu. All of the inputs indicated in the constructor have been set already, but you can override them here (for
        this run only) if you want.

        Returns: A dictionary containing all converged solutions obtained with your inputs.

        """
        # Set up a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            # Designate an intermediate file for file I/O
            output_filename = "output.txt"

            # Handle the airfoil file
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            # Handle the keystroke file
            keystrokes = self._default_keystrokes()
            keystrokes += [run_command]
            keystrokes += [
                "pwrt",
                f"{output_filename}",
                "y",
                "",
                "quit"
            ]

            ### Execute
            try:
                subprocess.run(
                    f'{self.xfoil_command} {airfoil_file}',
                    input="\n".join(keystrokes),
                    cwd=directory,
                            stdout=None if self.verbose else subprocess.DEVNULL,
                    text=True,
                    timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                pass

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
                try:
                    output_data = np.genfromtxt(
                        directory / output_filename,
                        skip_header=12,
                        usecols=np.arange(len(columns))
                    ).reshape(-1, len(columns))
                except OSError: # File not found
                    output_data = np.array([]).reshape(-1, len(columns))

            has_valid_inputs = len(output_data) != 0

            output_data_clean = {
                k: output_data[:, index] if has_valid_inputs else np.array([])
                for index, k in enumerate(columns)
            }

            return output_data_clean

    def alpha(self,
              alpha: Union[float, np.ndarray],
              start_at: Union[float, None] = 0,
              ) -> Dict[str, np.ndarray]:
        """
        Execute XFoil at a given angle of attack, or at a sequence of angles of attack.

        Args:

            alpha: The angle of attack [degrees]. Can be either a float or an iterable of floats, such as an array.

            start_at: Chooses whether to split a large sweep into two runs that diverge away from some central value,
            to improve convergence. As an example, if you wanted to sweep from alpha=-20 to alpha=20, you might want
            to instead do two sweeps and stitch them together: 0 to 20, and 0 to -20. `start_at` can be either:

                * None, in which case the alpha inputs are run as a single sequence in the order given.

                * A float that corresponds to an angle of attack (in degrees), in which case the alpha inputs are
                split into two sequences that diverge from the `start_at` value. Successful runs are then sorted by
                `alpha` before returning.

        Returns: A dictionary with the XFoil results. Dictionary values are arrays; they may not be the same shape as
        your input array if some points did not converge.

        """
        alphas = np.array(alpha).reshape(-1)

        if np.length(alphas) > 1:
            if start_at is not None:
                if start_at > np.min(alphas) and start_at < np.max(alphas):
                    alphas = np.sort(alphas)
                    alphas_upper = alphas[alphas > start_at]
                    alphas_lower = alphas[alpha <= start_at][::-1]

                    output = self._run_xfoil(
                        "\n".join(
                            [
                                f"a {a}"
                                for a in alphas_upper
                            ] + [
                                "init"
                            ] + [
                                f"a {a}"
                                for a in alphas_lower
                            ]
                        )
                    )

                    sort_order = np.argsort(output['alpha'])
                    output = {
                        k: v[sort_order]
                        for k, v in output.items()
                    }
                    return output

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
    # result_at_single_alpha = xf.alpha(5)
    # result_at_several_CLs = xf.cl([0.5, 0.7, 0.8, 0.9])
    # result_at_multiple_alphas = xf.alpha([3, 5, 60])  # Note: if a result does
    xf.alpha(np.arange(-5, 5))