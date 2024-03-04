from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airfoil
from typing import Union, List, Dict
import tempfile
import warnings
import os
import re


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

    # Defines an exception to throw if XFoil fails externally
    class XFoilError(Exception):
        pass

    def __init__(self,
                 airfoil: Airfoil,
                 Re: float = 0.,
                 mach: float = 0.,
                 n_crit: float = 9.,
                 xtr_upper: float = 1.,
                 xtr_lower: float = 1.,
                 hinge_point_x: float = 0.75,
                 full_potential: bool = False,
                 max_iter: int = 100,
                 xfoil_command: str = "xfoil",
                 xfoil_repanel: bool = True,
                 include_bl_data: bool = False,
                 verbose: bool = False,
                 timeout: Union[float, int, None] = 30,
                 working_directory: Union[Path, str] = None,
                 ):
        """
        Interface to XFoil. Compatible with both XFoil v6.xx (public) and XFoil v7.xx (private, contact Mark Drela at
        MIT for a copy.)

        Args:

            airfoil: The airfoil to analyze. Should be an AeroSandbox Airfoil object.

            Re: The chord-referenced Reynolds number. Set this to 0 to run in inviscid mode.

            mach: The freestream Mach number. Note that XFoil 6.xx uses the Karman-Tsien compressibility correction,
                which breaks down once supersonic flow is present (i.e., past M_crit). XFoil 7.xx has a full-potential
                solver that is theoretically-valid for weak shocks (perhaps up to M_crit + 0.05 or so).

            n_crit: The critical Tollmein-Schlichting wave amplification factor, as part of the "e^n" transition
                criterion. This is a measure of freestream turbulence and surface roughness. The following reference conditions
                are given in the XFoil documentation:

                - sailplane:                12-14
                - motorglider:              11-13
                - clean wind tunnel:        10-12
                - average wind tunnel:      9 (default)
                - dirty wind tunnel:        4-8

            xtr_upper: The upper-surface forced transition location [x/c], where the boundary layer will be
                automatically tripped to turbulent. Set to 1 to disable forced transition (default). Note that if the
                Reynolds number is sufficiently low, it is possible for the flow to re-laminarize after being tripped.

            xtr_lower: The lower-surface forced transition location [x/c], where the boundary layer will be
                automatically tripped to turbulent. Set to 1 to disable forced transition (default). Note that if the
                Reynolds number is sufficiently low, it is possible for the flow to re-laminarize after being tripped.

            hinge_point_x: The x/c location of the hinge point. This is used to calculate the hinge moment. If this is
                None, the hinge moment is not calculated.

            full_potential: If this is set True, it will turn full-potential mode on. Note that full-potential mode
                is only available in XFoil v7.xx or higher. (Unless you have specifically gone through the trouble of
                acquiring a copy of XFoil v7.xx you likely have v6.xx. Version 7.xx is not publicly distributed as of
                2023; contact Mark Drela at MIT for a copy.) Note that if you enable this flag with XFoil v6.xx,
                you'll likely get an error (no output file generated).

            max_iter: How many iterations should we let XFoil do?

            xfoil_command: The command-line argument to call XFoil, given as a string or a Path-like object.

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

            xfoil_repanel: Controls whether to allow XFoil to repanel your airfoil using its internal methods (PANE,
                with default settings, 160 nodes). Boolean, defaults to True.

            include_bl_data: Controls whether or not to include boundary layer data in the output. If this is True,
                the functions `alpha()` and `cl()` will return a dictionary with an additional key, "bl_data",
                which contains the boundary layer data in the form of a pandas DataFrame. Results in slightly higher
                runtime, mostly due to file I/O bottleneck. Defaults to False.

            verbose: Controls whether or not XFoil output is printed to command line. Defaults to False.

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
        self.hinge_point_x = hinge_point_x
        self.full_potential = full_potential
        self.max_iter = max_iter
        self.xfoil_command = xfoil_command
        self.xfoil_repanel = xfoil_repanel
        self.include_bl_data = include_bl_data
        self.verbose = verbose
        self.timeout = timeout

        if working_directory is None:
            self.working_directory = None
        else:
            self.working_directory = Path(working_directory)

    def __repr__(self):
        return f"XFoil(airfoil={self.airfoil}, Re={self.Re}, mach={self.mach}, n_crit={self.n_crit})"

    def _default_keystrokes(self,
                            airfoil_filename: str,
                            output_filename: str,
                            ) -> List[str]:
        """
        Returns a list of XFoil keystrokes that are common to all XFoil runs.

        Returns:
            A list of strings, each of which is a single XFoil keystroke to be followed by <enter>.
        """
        run_file_contents = []

        # Disable graphics
        run_file_contents += [
            "plop",
            "g",
            "w 0.05",
            "",
        ]

        # Load the airfoil
        run_file_contents += [
            f"load {airfoil_filename}",
        ]

        if self.xfoil_repanel:
            run_file_contents += [
                "ppar",
                "n 279"  # Highest number of panel points before XFoil IWX array overfills and starts trimming wake
                "",
                "",
                "",
                # "pane",
            ]

        # Enter oper mode
        run_file_contents += [
            "oper",
        ]

        # Handle Re
        if self.Re != 0:
            run_file_contents += [
                f"v {self.Re:.8g}",
            ]

        # Handle mach
        run_file_contents += [
            f"m {self.mach:.8g}",
        ]

        # Handle hinge moment
        run_file_contents += [
            "hinc",
            f"fnew {float(self.hinge_point_x):.8g} {float(self.airfoil.local_camber(self.hinge_point_x)):.8g}",
            "fmom",
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
        if not (self.xtr_upper == 1 and self.xtr_lower == 1 and self.n_crit == 9):
            run_file_contents += [
                "vpar",
                f"xtr {self.xtr_upper:.8g} {self.xtr_lower:.8g}",
                f"n {self.n_crit:.8g}",
                "",
            ]

        # Set polar accumulation
        run_file_contents += [
            "pacc",
            f"{output_filename}",
            "",
        ]

        # Include more data in polar
        run_file_contents += [
            "cinc"  # include minimum Cp
        ]

        return run_file_contents

    def _run_xfoil(self,
                   run_command: str,
                   read_bl_data_from: str = None,
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
            keystrokes = self._default_keystrokes(
                airfoil_filename=airfoil_file,
                output_filename=output_filename
            )
            keystrokes += [run_command]
            keystrokes += [
                "pacc",  # End polar accumulation
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
                # command = f'{self.xfoil_command} {airfoil_file}' # Old syntax; try this if calls are not working
                proc = subprocess.Popen(
                    self.xfoil_command,
                    cwd=directory,
                    stdin=subprocess.PIPE,
                    stdout=None if self.verbose else subprocess.DEVNULL,
                    stderr=None if self.verbose else subprocess.DEVNULL,
                    text=True,
                    # shell=True,
                    # timeout=self.timeout,
                    # check=True
                )
                outs, errs = proc.communicate(
                    input="\n".join(keystrokes),
                    timeout=self.timeout
                )
                return_code = proc.poll()

            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()

                warnings.warn(
                    "XFoil run timed out!\n"
                    "If this was not expected, try increasing the `timeout` parameter\n"
                    "when you create this AeroSandbox XFoil instance.",
                    stacklevel=2
                )
            except subprocess.CalledProcessError as e:
                if e.returncode == 11:
                    raise self.XFoilError(
                        "XFoil segmentation-faulted. This is likely because your input airfoil has too many points.\n"
                        "Try repaneling your airfoil with `Airfoil.repanel()` before passing it into XFoil.\n"
                        "For further debugging, turn on the `verbose` flag when creating this AeroSandbox XFoil instance.")
                elif e.returncode == 8 or e.returncode == 136:
                    raise self.XFoilError(
                        "XFoil returned a floating point exception. This is probably because you are trying to start\n"
                        "your analysis at an operating point where the viscous boundary layer can't be initialized based\n"
                        "on the computed inviscid flow. (You're probably hitting a Goldstein singularity.) Try starting\n"
                        "your XFoil run at a less-aggressive (alpha closer to 0, higher Re) operating point.")
                elif e.returncode == 1:
                    raise self.XFoilError(
                        f"Command '{self.xfoil_command}' returned non-zero exit status 1.\n"
                        f"This is likely because AeroSandbox does not see XFoil on PATH with the given command.\n"
                        f"Check the logs (`asb.XFoil(..., verbose=True)`) to verify that this is the case, and if so,\n"
                        f"provide the correct path to the XFoil executable in the asb.XFoil constructor via `xfoil_command=`."
                    )
                else:
                    raise e

            ### Parse the polar
            try:
                with open(directory / output_filename) as f:
                    lines = f.readlines()
            except FileNotFoundError:
                raise self.XFoilError(
                    "It appears XFoil didn't produce an output file, probably because it crashed.\n"
                    "To troubleshoot, try some combination of the following:\n"
                    "\t - In the XFoil constructor, verify that either XFoil is on PATH or that the `xfoil_command` parameter is set.\n"
                    "\t - In the XFoil constructor, run with `verbose=True`.\n"
                    "\t - In the XFoil constructor, set the `working_directory` parameter to a known folder to see the XFoil input and output files.\n"
                    "\t - In the XFoil constructor, set the `timeout` parameter to a large number to see if XFoil is just taking a long time to run.\n"
                    "\t - On Windows, use `XFoil.open_interactive()` to run XFoil interactively in a new window.\n"
                    "\t - Try allowing XFoil to repanel the airfoil by setting `xfoil_repanel=True` in the XFoil constructor.\n"
                )

            try:
                separator_line = None
                for i, line in enumerate(lines):
                    # The first line with at least 30 "-" in it is the separator line.
                    if line.count("-") >= 30:
                        separator_line = i
                        break

                if separator_line is None:
                    raise IndexError

                title_line = lines[i - 1]
                columns = title_line.split()

                data_lines = lines[i + 1:]

            except IndexError:
                raise self.XFoilError(
                    "XFoil output file is malformed; it doesn't have the expected number of lines.\n"
                    "For debugging, the raw output file from XFoil is printed below:\n"
                    + "\n".join(lines)
                    + "\nTitle line: " + title_line
                    + "\nColumns: " + str(columns)
                )

            def str_to_float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return np.nan

            output = {
                column: []
                for column in [
                    "alpha",
                    "CL",
                    "CD",
                    "CDp",
                    "CM",
                    "Cpmin",
                    "Xcpmin",
                    "Chinge",
                    "Top_Xtr",
                    "Bot_Xtr",
                ]
            }

            for pointno, line in enumerate(data_lines):
                float_pattern = r"-?\d+\.\d+"
                entries = re.findall(float_pattern, line)
                data = [str_to_float(entry) for entry in entries]

                if len(data) == 10 and len(columns) == 8:
                    # This is a monkey-patch for a bug in XFoil v6.99, which causes polar output files to be malformed
                    # when including both Cpmin ("cinc") and hinge moment ("hinc") in the same run.
                    columns = [
                        "alpha",
                        "CL",
                        "CD",
                        "CDp",
                        "CM",
                        "Cpmin",
                        "Xcpmin",
                        "Chinge",
                        "Top_Xtr",
                        "Bot_Xtr",
                    ]

                if not len(data) == len(columns):
                    raise self.XFoilError(
                        "XFoil output file is malformed; the header and data have different numbers of columns.\n"
                        "In previous testing, this occurs due to a bug in XFoil itself, with certain input combos.\n"
                        "For debugging, the raw output file from XFoil is printed below:\n"
                        + "\n".join(lines)
                        + "\nTitle line: " + title_line
                        + f"\nIdentified {len(data)} data columns and {len(columns)} header columns."
                        + "\nColumns: " + str(columns)
                        + "\nData: " + str(data)
                    )

                for i in range(len(columns)):
                    output[columns[i]].append(data[i])

            output = {
                k: np.array(v, dtype=float)
                for k, v in output.items()
            }

            # Read the BL data
            if read_bl_data_from is not None:
                import pandas as pd
                bl_datas: List[pd.DataFrame] = []

                if read_bl_data_from == "alpha":
                    alpha_to_dump_mapping = {
                        float(dump_filename.stem.split("_")[-1]): dump_filename
                        for dump_filename in directory.glob("dump_a_*.txt")
                    }

                    for alpha in output["alpha"]:
                        dump_filename = alpha_to_dump_mapping[
                            min(alpha_to_dump_mapping.keys(), key=lambda x: abs(x - alpha))
                        ]

                        bl_datas.append(
                            pd.read_csv(
                                dump_filename,
                                sep="\s+",
                                names=["s", "x", "y", "ue/vinf", "dstar", "theta", "cf", "H"],
                                skiprows=1,
                            )
                        )

                elif read_bl_data_from == "cl":
                    cl_to_dump_mapping = {
                        float(dump_filename.stem.split("_")[-1]): dump_filename
                        for dump_filename in directory.glob("dump_cl_*.txt")
                    }

                    for cl in output["CL"]:
                        dump_filename = cl_to_dump_mapping[
                            min(cl_to_dump_mapping.keys(), key=lambda x: abs(x - cl))
                        ]

                        bl_datas.append(
                            pd.read_csv(
                                dump_filename,
                                sep="\s+",
                                names=["s", "x", "y", "ue/vinf", "dstar", "theta", "cf", "H"],
                                skiprows=1,
                            )
                        )

                else:
                    raise ValueError("The `read_bl_data_from` parameter must be 'alpha', 'cl', or None.")

                # Augment the output data for each BL
                for bl_data in bl_datas:
                    # Get Cp via Karman-Tsien compressibility correction, same as XFoil
                    Cp_0 = (1 - bl_data["ue/vinf"] ** 2)
                    bl_data["Cp"] = (Cp_0 /
                                     (
                                             np.sqrt(1 - self.mach ** 2)
                                             + (
                                                     (self.mach ** 2)
                                                     / (1 + np.sqrt(1 - self.mach ** 2))
                                                     * (Cp_0 / 2)
                                             )

                                     )
                                     )

                    # Get Re_theta
                    bl_data["Re_theta"] = np.abs(bl_data["ue/vinf"]) * bl_data["theta"] * self.Re

                output["bl_data"] = np.fromiter(bl_datas, dtype="O")

            return output

    def open_interactive(self) -> None:
        """
        Opens a new terminal window and runs XFoil interactively. This is useful for detailed analysis or debugging.

        Returns: None
        """
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            ### Handle the airplane file
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            ### Open up AVL
            import sys, os
            if sys.platform == "win32":
                # Run XFoil
                print("Running XFoil interactively in a new window, quit it to continue...")

                command = f'cmd /k "{self.xfoil_command} {airfoil_file}"'

                process = subprocess.Popen(
                    command,
                    cwd=directory,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                process.wait()

            else:
                raise NotImplementedError(
                    "Ability to auto-launch interactive XFoil sessions isn't yet implemented for non-Windows OSes."
                )

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
        alphas = np.reshape(np.array(alpha), -1)
        alphas = np.sort(alphas)

        commands = []

        def schedule_run(alpha: float):

            commands.append(f"a {alpha}")

            if self.hinge_point_x is not None:
                commands.append("fmom")

            if self.include_bl_data:
                commands.extend([
                    f"dump dump_a_{alpha:.8f}.txt",
                    # "vplo",
                    # "cd", # Dissipation coefficient
                    # f"dump cdis_a_{alpha:.8f}.txt",
                    # f"n", # Amplification ratio
                    # f"dump n_a_{alpha:.8f}.txt",
                    # "",
                ])

        if (
                len(alphas) > 1 and
                (start_at is not None) and
                (np.min(alphas) < start_at < np.max(alphas))
        ):
            alphas_upper = alphas[alphas > start_at]
            alphas_lower = alphas[alpha <= start_at][::-1]

            for a in alphas_upper:
                schedule_run(a)

            commands.append("init")

            for a in alphas_lower:
                schedule_run(a)
        else:
            for a in alphas:
                schedule_run(a)

        output = self._run_xfoil(
            "\n".join(commands),
            read_bl_data_from="alpha" if self.include_bl_data else None
        )

        sort_order = np.argsort(output['alpha'])
        output = {
            k: v[sort_order]
            for k, v in output.items()
        }
        return output

    def cl(self,
           cl: Union[float, np.ndarray],
           start_at: Union[float, None] = 0,
           ) -> Dict[str, np.ndarray]:
        """
        Execute XFoil at a given lift coefficient, or at a sequence of lift coefficients.

        Args:
            cl: The lift coefficient [-]. Can be either a float or an iterable of floats, such as an array.

            start_at: Chooses whether to split a large sweep into two runs that diverge away from some central value,
            to improve convergence. As an example, if you wanted to sweep from cl=-1.5 to cl=1.5, you might want to
            instead do two sweeps and stitch them together: 0 to 1.5, and 0 to -1.5. `start_at` can be either:

                * None, in which case the cl inputs are run as a single sequence in the order given.

                * A float that corresponds to an lift coefficient, in which case the cl inputs are
                split into two sequences that diverge from the `start_at` value. Successful runs are then sorted by
                `alpha` before returning.


        Returns: A dictionary with the XFoil results. Dictionary values are arrays; they may not be the same shape as
        your input array if some points did not converge.

        """
        cls = np.reshape(np.array(cl), -1)
        cls = np.sort(cls)

        commands = []

        def schedule_run(cl: float):

            commands.append(f"cl {cl}")

            if self.hinge_point_x is not None:
                commands.append("fmom")

            if self.include_bl_data:
                commands.extend([
                    f"dump dump_cl_{cl:.8f}.txt",
                    # "vplo",
                    # "cd", # Dissipation coefficient
                    # f"dump cdis_cl_{cl:.8f}.txt",
                    # f"n", # Amplification ratio
                    # f"dump n_cl_{cl:.8f}.txt",
                    # "",
                ])

        if (
                len(cls) > 1 and
                (start_at is not None) and
                (np.min(cls) < start_at < np.max(cls))
        ):
            cls_upper = cls[cls > start_at]
            cls_lower = cls[cls <= start_at][::-1]

            for c in cls_upper:
                schedule_run(c)

            commands.append("init")

            for c in cls_lower:
                schedule_run(c)
        else:
            for c in cls:
                schedule_run(c)

        output = self._run_xfoil(
            "\n".join(commands),
            read_bl_data_from="cl" if self.include_bl_data else None
        )

        sort_order = np.argsort(output['alpha'])
        output = {
            k: v[sort_order]
            for k, v in output.items()
        }
        return output


if __name__ == '__main__':
    af = Airfoil("naca2412").repanel(n_points_per_side=100)

    xf = XFoil(
        airfoil=af,
        Re=1e6,
        hinge_point_x=0.75,
        include_bl_data=True,
        working_directory=str(Path.home() / "Downloads" / "test"),
    )

    result_at_single_alpha = xf.alpha(5)

    result_at_several_CLs = xf.cl([-0.1, 0.5, 0.7, 0.8, 0.9])

    result_at_multiple_alphas = xf.alpha([3, 5, 60])
    # Note: if a result does not converge (such as the 60 degree case here), it will not be included in the results.
