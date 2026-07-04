from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import ConcreteVectorizable
from typing import Literal
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airfoil
from aerosandbox.aerodynamics.aero_3D.avl import AVL
import tempfile
from textwrap import dedent


class MSES(ExplicitAnalysis):
    """

    An interface to MSES, MSET, and MPLOT, a 2D airfoil analysis system developed by Mark Drela at MIT.

    Requires compiled binaries for all the programs to be on your computer;
    MSES is available here: https://web.mit.edu/drela/Public/web/mses/
    Academics can get a copy by emailing the MIT Tech. Licensing Office;
    MIT affiliates can find a copy on Athena.

    It is recommended (but not required) that you add MSES, MSET, and MPLOT to your system PATH environment variable
    such that they can be called with the commands `mses`, `mset`, and `mplot`. If this is not the case, you need to
    specify the path to these executables using the command arguments of the constructor.

    -----
    X11 Notes:

    Note that MSES, MSET, and MPLOT by default open up X11 windows on your computer. If you prefer that this doesn't
    happen (for extra speed), or if you cannot have this happen (e.g., you are computing in an environment without
    proper X11 support, like Windows Subsystem for Linux), you should use XVFB. https://en.wikipedia.org/wiki/Xvfb

    XVFB is a virtual "display" server that can receive X11 output and safely dump it. (If you don't use XVFB and you
    don't have proper X11 support on your computer, this AeroSandbox MSES module will simply error out during the
    MSET call - probably not what you want.)

    To install XVFB on a Linux machine, use:

    ```bash
    sudo apt-get install xvfb
    ```

    Then, when instantiating this MSES instance in AeroSandbox, pass the `use_xvfb` flag to be True. Default behavior
    here is that this class will look for the XVFB executable, `xvfb-run`, on your machine. If it finds it,
    it will run with XVFB enabled. If it does not, it will run without XVFB.

    -----

    Usage example:

    >>> ms = MSES(
    >>>     airfoil=Airfoil("naca2412").repanel(n_points_per_side=100),
    >>> )
    >>>
    >>> result_at_single_operating_point = ms.run(alpha=5, Re=1e6, mach=0.2)
    >>> result_at_multiple_alphas = ms.run(alpha=[3, 5, 60], Re=1e6, mach=0.2)
    >>> # Note: if a run does not converge (such as the 60 degree case here), it will not be included in the results.

    """

    def __init__(
        self,
        airfoil: Airfoil,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        max_iter: int = 100,
        mset_command: str = "mset",
        mses_command: str = "mses",
        mplot_command: str = "mplot",
        use_xvfb: bool | None = None,
        xvfb_command: str = "xvfb-run -a",
        verbosity: int = 1,
        timeout_mset: float | int | None = 10,
        timeout_mses: float | int | None = 60,
        timeout_mplot: float | int | None = 10,
        working_directory: str | None = None,
        behavior_after_unconverged_run: Literal[
            "reinitialize", "terminate"
        ] = "reinitialize",
        mset_alpha: float = 0,
        mset_n: int = 141,
        mset_e: float = 0.4,
        mset_io: int = 37,
        mset_x: float = 0.850,
        mses_mcrit: float = 0.99,
        mses_mucon: float = -1.0,
    ):
        """
        Interface to MSES, MSET, and MPLOT, a 2D airfoil analysis system developed by Mark Drela at MIT.

        Args:
            airfoil: The airfoil to analyze. Should be an AeroSandbox Airfoil object.

            n_crit: The critical Tollmein-Schlichting wave amplification factor, as part of the "e^n" transition
                criterion. This is a measure of freestream turbulence and surface roughness.

            xtr_upper: The upper-surface forced transition location [x/c], where the boundary layer will be
                automatically tripped to turbulent. Set to 1 to disable forced transition (default).

            xtr_lower: The lower-surface forced transition location [x/c], where the boundary layer will be
                automatically tripped to turbulent. Set to 1 to disable forced transition (default).

            max_iter: How many iterations should we let MSES do?

            mset_command: The command-line argument to call MSET.

                * If MSET is on your system PATH, then you can just leave this as "mset".

                * If MSET is not on your system PATH, then you should provide a filepath to the MSET executable.

            mses_command: The command-line argument to call MSES. (Same rules as `mset_command`.)

            mplot_command: The command-line argument to call MPLOT. (Same rules as `mset_command`.)

            use_xvfb: Controls whether Xvfb is used to soak up the X11 windows opened by MSET, MSES, and MPLOT.
                (See the "X11 Notes" in the class docstring above.) If this is None (default), Xvfb is used if
                and only if the `xvfb_command` executable appears to be available on this machine.

            xvfb_command: The command-line argument to call Xvfb, if it is used.

            verbosity: Controls the amount of printout. Set to 0 for silent operation; 1 (default) prints
                high-level progress; 2 also prints the raw MSET / MSES / MPLOT outputs.

            timeout_mset: Controls how long any individual MSET run is allowed to take before the process is
                killed. Given in units of seconds. To disable the timeout, set this to None.

            timeout_mses: Controls how long any individual MSES run is allowed to take before the process is
                killed. Given in units of seconds. To disable the timeout, set this to None.

            timeout_mplot: Controls how long any individual MPLOT run is allowed to take before the process is
                killed. Given in units of seconds. To disable the timeout, set this to None.

            working_directory: Controls which working directory is used for the input and output files. By
                default, this is set to a TemporaryDirectory that is deleted after the run. However, you can set
                it to somewhere local for debugging purposes.

            behavior_after_unconverged_run: Controls what happens when a run in a (multipoint) sweep does not
                converge. Options are:

                * "reinitialize" (default): Reinitializes the mesh at the next operating point and continues the
                    sweep.

                * "terminate": Skips all subsequent runs; any already-converged results are still returned.

            mset_alpha: The angle of attack [degrees] used when generating the initial mesh with MSET. (Currently
                unused; the mesh is instead initialized at the first angle of attack in the run sweep.)

            mset_n: The number of airfoil-surface gridpoints; the value passed to option "n" of MSET's
                grid-parameter menu.

            mset_e: The value passed to option "e" (grid exponent) of MSET's grid-parameter menu.

            mset_io: The value passed to options "i" and "o" (inlet/outlet streamwise gridpoints) of MSET's
                grid-parameter menu.

            mset_x: The value passed to option "x" of MSET's grid-parameter menu.

            mses_mcrit: The critical Mach number ("MCRIT") written to the MSES case file. Governs where MSES
                begins to add artificial dissipation.

            mses_mucon: The artificial dissipation constant ("MUCON") written to the MSES case file.

        """
        if use_xvfb is None:
            trial_run = subprocess.run(
                xvfb_command,  # Analogous to "xvfb-run", perhaps with additional arguments
                capture_output=True,
                shell=True,
                text=True,
            )
            expected_result = "xvfb-run: usage error:"
            use_xvfb = (
                expected_result in trial_run.stderr
                or expected_result in trial_run.stdout
            )

        if not use_xvfb:
            xvfb_command = ""

        self.airfoil = airfoil
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.max_iter = max_iter
        self.mset_command = mset_command
        self.mses_command = mses_command
        self.mplot_command = mplot_command
        self.use_xvfb = use_xvfb
        self.xvfb_command = xvfb_command
        self.verbosity = verbosity
        self.timeout_mses = timeout_mses
        self.timeout_mset = timeout_mset
        self.timeout_mplot = timeout_mplot
        self.working_directory = working_directory
        self.behavior_after_unconverged_run = behavior_after_unconverged_run
        self.mset_alpha = mset_alpha
        self.mset_n = mset_n
        self.mset_e = mset_e
        self.mset_io = mset_io
        self.mset_x = mset_x
        self.mses_mcrit = mses_mcrit
        self.mses_mucon = mses_mucon

    def run(
        self,
        alpha: ConcreteVectorizable = 0.0,
        Re: ConcreteVectorizable = 0.0,
        mach: ConcreteVectorizable = 0.01,
    ):
        """
        Runs MSES at one or more operating points and returns the results.

        The `alpha`, `Re`, and `mach` inputs are broadcast against each other (in the NumPy sense) to form the
        sequence of operating points, which are then run in order.

        Args:
            alpha: The angle of attack [degrees]. Can be either a float or an iterable of floats, such as an array.

            Re: The chord-referenced Reynolds number. Can be either a float or an iterable of floats, such as an
                array.

            mach: The freestream Mach number. Can be either a float or an iterable of floats, such as an array.

        Returns:
            A dictionary with the MSES results, where each value is a numpy array with one entry per converged
            run. Keys are parsed from the MPLOT output and include "mach", "alpha", "CL", "CD", etc. Runs that do
            not converge are not included. If no runs converge, an empty dictionary is returned.

        """
        ### Make all inputs iterables:
        alphas, Res, machs = np.broadcast_arrays(
            np.ravel(alpha),
            np.ravel(Re),
            np.ravel(mach),
        )

        # Set up a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            # Handle the airfoil file
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            def mset(mset_alpha):
                mset_keystrokes = dedent(
                    f"""\
                15
                case
                7
                n {self.mset_n}
                e {self.mset_e}
                i {self.mset_io}
                o {self.mset_io}
                x {self.mset_x}
                
                1
                {mset_alpha}
                2
                
                3
                4
                0
                """
                )

                if self.verbosity >= 1:
                    print(f"Generating mesh at alpha = {mset_alpha} with MSES...")

                return subprocess.run(
                    f'{self.xvfb_command} "{self.mset_command}" "{airfoil_file}"',
                    input=mset_keystrokes,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                    timeout=self.timeout_mset,
                )

            try:
                mset(mset_alpha=alphas[0])
            except subprocess.CalledProcessError as e:
                print(e.stdout)
                print(e.stderr)
                if "BadName (named color or font does not exist)" in e.stderr:
                    raise RuntimeError(
                        "MSET via AeroSandbox errored because it couldn't launch an X11 window.\n"
                        "Try either installing a typical X11 client, or install Xvfb, which is\n"
                        "a virtual X11 server. More details in the AeroSandbox MSES docstring."
                    )
                else:
                    raise

            runs_output = {}

            for i, (alpha, mach, Re) in enumerate(zip(alphas, machs, Res)):
                if self.verbosity >= 1:
                    print(
                        f"Solving alpha = {alpha:.3f}, mach = {mach:.4f}, Re = {Re:.3e} with MSES..."
                    )

                with open(directory / "mses.case", "w+") as f:
                    f.write(
                        dedent(
                            f"""\
                    3  4  5  7
                    3  4  5  7
                    {mach}   0.0   {alpha} | MACHin  CLIFin  ALFAin
                    3  2                             | ISMOM  IFFBC  [ DOUXin DOUYin SRCEin ]
                    {Re}  {self.n_crit}          | REYNin ACRIT [ KTRTYP ]
                    {self.xtr_lower}    {self.xtr_upper}                   | XTR1 XTR2
                    {self.mses_mcrit}  {self.mses_mucon}                      | MCRIT  MUCON
                    0    0                           | ISMOVE  ISPRES
                    0    0                           | NMODN   NPOSN
                    
                    """
                        )
                    )

                mses_keystrokes = dedent(
                    f"""\
                    {self.max_iter}
                    0
                    """
                )

                mses_run = subprocess.run(
                    f'{self.xvfb_command} "{self.mses_command}" case',
                    input=mses_keystrokes,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                    timeout=self.timeout_mses,
                )
                if self.verbosity >= 2:
                    print(mses_run.stdout)
                    print(mses_run.stderr)

                converged = "Converged on tolerance" in mses_run.stdout
                if not converged:
                    if self.behavior_after_unconverged_run == "reinitialize":
                        if self.verbosity >= 1:
                            print(
                                "Run did not converge. Reinitializing mesh and continuing..."
                            )
                        try:
                            next_alpha = alphas[i + 1]
                        except IndexError:
                            break
                        mset(mset_alpha=next_alpha)
                    elif self.behavior_after_unconverged_run == "terminate":
                        if self.verbosity >= 1:
                            print(
                                "Run did not converge. Skipping all subsequent runs..."
                            )
                        break

                    continue

                mplot_keystrokes = dedent(
                    """\
                        1
                        12
                        0
                        0
                    """
                )

                mplot_run = subprocess.run(
                    f'{self.xvfb_command} "{self.mplot_command}" case',
                    input=mplot_keystrokes,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                    timeout=self.timeout_mplot,
                )
                if self.verbosity >= 2:
                    print(mplot_run.stdout)
                    print(mplot_run.stderr)

                raw_output = (
                    mplot_run.stdout.replace("top Xtr", "xtr_top")
                    .replace("bot Xtr", "xtr_bot")
                    .replace("at x,y", "x_ac")
                )

                run_output = AVL.parse_unformatted_data_output(raw_output)

                # Merge runs_output and run_output
                for k in run_output.keys():
                    try:
                        runs_output[k].append(run_output[k])
                    except KeyError:  # List not created yet
                        runs_output[k] = [run_output[k]]

            # Clean up the dictionary
            runs_output = {k: np.array(v) for k, v in runs_output.items()}

            if len(runs_output) == 0:  # No runs converged
                return runs_output

            # runs_output["mach"] = runs_output.pop("Ma")
            runs_output = {"mach": runs_output.pop("Ma"), **runs_output}

            return runs_output


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    ms = MSES(
        airfoil=Airfoil("rae2822"),  # .repanel(n_points_per_side=30),
        working_directory="/mnt/c/Users/peter/Downloads/msestest/",
        # max_iter=120,
        verbosity=1,
        behavior_after_unconverged_run="terminate",
        mset_n=300,
        max_iter=100,
        # verbose=False
    )
    res = ms.run(
        alpha=3,
        mach=np.arange(0.55, 0.8, 0.005),
        # Re=1e6,
    )
    pprint(res)

    import matplotlib

    matplotlib.use("WebAgg")

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(res["mach"], res["CD"], ".-")
    p.show_plot()
