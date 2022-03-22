from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airfoil
from typing import Union, List, Dict
import tempfile
import warnings
import os


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
                 full_potential: bool = False,
                 max_iter: int = 100,
                 xfoil_command: str = "xfoil",
                 xfoil_repanel: bool = True,
                 verbose: bool = False,
                 timeout: Union[float, int, None] = 30,
                 working_directory: str = None,
                 ):
        """
        Interface to XFoil. Compatible with both XFoil v6.xx (public) and XFoil v7.xx (private, contact Mark Drela at
        MIT for a copy.)

        Args:

            airfoil: The angle of attack [degrees]

            Re: The chord-referenced Reynolds number

            mach: The freestream Mach number

            n_crit: The critical Tollmein-Schlichting wave amplification factor

            xtr_upper: The upper-surface trip location [x/c]

            xtr_lower: The lower-surface trip location [x/c]

            full_potential: If this is set True, it will turn full-potential mode on. Note that full-potential mode
            is only available in XFoil v7.xx or higher. (Unless you have specifically gone through the trouble of
            acquiring a copy of XFoil v7.xx you likely have v6.xx. Version 7.xx is not publicly distributed as of
            2022; contact Mark Drela at MIT for a copy.) Note that if you enable this flag with XFoil v6.xx,
            you'll likely get an error (no output file generated).

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
        if mach >= 1:
            raise ValueError("XFoil will terminate if a supersonic freestream Mach number is given.")

        self.airfoil = airfoil
        self.Re = Re
        self.mach = mach
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.full_potential = full_potential
        self.max_iter = max_iter
        self.xfoil_command = xfoil_command
        self.xfoil_repanel = xfoil_repanel
        self.verbose = verbose
        self.timeout = timeout
        self.working_directory = working_directory

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

        if self.full_potential:
            run_file_contents += [
                "full",
                "fpar",
                f"i {self.max_iter}",
                "",
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

            # Remove an old output file, if one exists:
            try:
                os.remove(directory / output_filename)
            except FileNotFoundError:
                pass

            ### Execute
            try:
                subprocess.run(
                    f'{self.xfoil_command} {airfoil_file}',
                    input="\n".join(keystrokes),
                    cwd=directory,
                    stdout=None if self.verbose else subprocess.DEVNULL,
                    stderr=None if self.verbose else subprocess.DEVNULL,
                    text=True,
                    shell=True,
                    timeout=self.timeout,
                    check=True
                )
            except subprocess.TimeoutExpired:
                warnings.warn(
                    "XFoil run timed out!\n"
                    "If this was not expected, try increasing the `timeout` parameter\n"
                    "when you create this AeroSandbox XFoil instance.",
                    stacklevel=2
                )
            except subprocess.CalledProcessError as e:
                if e.returncode == 11:
                    raise RuntimeError(
                        "XFoil segmentation-faulted. This is likely because your input airfoil has too many points.\n"
                        "Try repaneling your airfoil with `Airfoil.repanel()` before passing it into XFoil.\n"
                        "For further debugging, turn on the `verbose` flag when creating this AeroSandbox XFoil instance.")
                elif e.returncode == 8 or e.returncode == 136:
                    raise RuntimeError(
                        "XFoil returned a floating point exception. This is probably because you are trying to start\n"
                        "your analysis at an operating point where the viscous boundary layer can't be initialized based\n"
                        "on the computed inviscid flow. (You're probably hitting a Goldstein singularity.) Try starting\n"
                        "your XFoil run at a less-aggressive operating point.")
                else:
                    raise e

            ### Parse the polar
            try:
                with open(directory / output_filename) as f:
                    lines = f.readlines()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "It appears XFoil didn't produce an output file, probably because it crashed.\n"
                    "Try running with `verbose=True` in the XFoil constructor to see what's going on."
                )

            title_line = lines[10]
            columns = title_line.split()

            output = {
                column: []
                for column in columns
            }

            def str_to_float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return np.NaN

            for line in lines[12:]:
                data = [str_to_float(entry) for entry in line.split()]
                for i in range(len(columns)):
                    output[columns[i]].append(data[i])

            output = {
                k: np.array(v, dtype=float)
                for k, v in output.items()
            }

            return output

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
                if np.min(alphas) < start_at < np.max(alphas):
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
