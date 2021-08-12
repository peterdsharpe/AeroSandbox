from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airplane
from aerosandbox.performance import OperatingPoint
from typing import Union, List, Dict
import tempfile
import warnings


class AVL(ExplicitAnalysis):
    """

    An interface to AVL, a 3D vortex lattice aerodynamics code developed by Mark Drela at MIT.

    Requires AVL to be on your computer; AVL is available here: https://web.mit.edu/drela/Public/web/avl/

    It is recommended (but not required) that you add AVL to your system PATH environment variable such that it can
    be called with the command `avl`. If this is not the case, you need to specify the path to your AVL
    executable using the `avl_command` argument of the constructor.

    Usage example:

        >>>avl = asb.AVL(
        >>>    airplane=my_airplane,
        >>>    op_point=asb.OperatingPoint(
        >>>        velocity=100, # m/s
        >>>        alpha=5, # deg
        >>>        beta=4, # deg
        >>>        p=0.01, # rad/sec
        >>>        q=0.02, # rad/sec
        >>>        r=0.03, # rad/sec
        >>>    )
        >>>)
        >>>outputs = avl.run()

    """

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint = OperatingPoint(),
                 avl_command: str = "avl",
                 verbose: bool = False,
                 working_directory: str = None,
                 ):
        """
        Interface to AVL.

        Args:

            airplane: The airplane object you wish to analyze.

            op_point: The operating point you wish to analyze at.

            avl_command: The command-line argument to call AVL.

                * If AVL is on your system PATH, then you can just leave this as "avl".

                * If AVL is not on your system PATH, thjen you should provide a filepath to the AVL executable.

                Note that AVL is not on your PATH by default. To tell if AVL is on your system PATH, open up a
                terminal and type "avl".

                    * If the AVL menu appears, it's on your PATH.

                    * If you get something like "'avl' is not recognized as an internal or external command..." or
                    "Command 'avl' not found, did you mean...", then it is not on your PATH and you'll need to
                    specify the location of your AVL executable as a string.

                To add AVL to your path, modify your system's environment variables. (Google how to do this for your OS.)

            verbose:

            working_directory:
        """
        self.airplane = airplane
        self.op_point = op_point
        self.avl_command = avl_command
        self.verbose = verbose
        self.working_directory = working_directory

    def run(self) -> Dict:
        return self._run_avl()

    def _default_keystroke_file_contents(self) -> List[str]:

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

        # Set parameters
        run_file_contents += [
            "m"
            f"mn {self.op_point.mach()}",
            f"v {self.op_point.velocity}",
            f"d {self.op_point.atmosphere.density()}",
            "g 9.81",
            ""
        ]

        # Set analysis state
        p_bar = self.op_point.p * self.airplane.b_ref / (2 * self.op_point.velocity)
        q_bar = self.op_point.q * self.airplane.c_ref / (2 * self.op_point.velocity)
        r_bar = self.op_point.r * self.airplane.b_ref / (2 * self.op_point.velocity)

        run_file_contents += [
            f"a a {self.op_point.alpha}",
            f"b b {self.op_point.beta}",
            f"r r {p_bar}",
            f"p p {q_bar}",
            f"y y {r_bar}"
        ]

        return run_file_contents

    def _run_avl(self,
                 run_command: str = None,
                 ) -> Dict[str, np.ndarray]:
        """
        Private function to run AVL.

        Args: run_command: A string with any AVL keystroke inputs that you'd like. By default, you start off within the OPER
        menu. All of the inputs indicated in the constructor have been set already, but you can override them here (
        for this run only) if you want.

        Returns: A dictionary containing all of your results.

        """
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            # Designate an intermediate file for file I/O
            output_filename = "output.txt"
            with open(directory / output_filename, "w+") as f:
                pass

            # Handle the airplane file
            airplane_file = "airplane.avl"
            self.airplane.write_avl(directory / airplane_file)

            # Handle the run file
            keystroke_file_contents = self._default_keystroke_file_contents()
            if run_command is not None:
                keystroke_file_contents += [run_command]
            keystroke_file_contents += [
                "x",
                "st",
                f"{output_filename}",
                "o",
                "",
                "",
                "quit"
            ]
            keystroke_file = "keystroke_file.txt"
            with open(directory / keystroke_file, "w+") as f:
                f.write(
                    "\n".join(keystroke_file_contents)
                )

            command = f'{self.avl_command} {airplane_file} < {keystroke_file}'

            ### Execute
            subprocess.call(
                command,
                shell=True,
                cwd=directory,
                stdout=None if self.verbose else subprocess.DEVNULL
            )

            ##### Parse the output file
            # Read the file
            with open(directory / output_filename, "r") as f:
                output_data = f.read()

            # Trim off the first few lines that contain name, # of panels, etc.
            output_data = "\n".join(output_data.split("\n")[8:])

            ### Iterate through the string to find all the numeric values, based on where "=" appears.
            values = []
            index = output_data.find("=")
            while index != -1:
                output_data = output_data[index + 1:]
                number = output_data[:12].split("\n")[0]
                number = float(number)
                values.append(number)

                index = output_data.find("=")

            ### Record the keys associated with those values:
            keys = [
                "Sref",
                "Cref",
                "Bref",
                "Xref",
                "Yref",
                "Zref",
                "alpha",
                "pb/2V",
                "p'b/2V",
                "beta",
                "qc/2V",
                "mach",
                "rb/2V",
                "r'b/2V",
                "CX",  # Note: these refer to "CXtot", etc. in AVL, but the "tot" is redundant.
                "Cl",
                "Cl'",
                "CY",
                "Cm",
                "CZ",
                "Cn",
                "Cn'",
                "CL",
                "CD",
                "CDvis",
                "CDind",
                "CLff",
                "CDff",
                "Cyff",
                "e",
                "CLa",
                "CLb",
                "CYa",
                "CYb",
                "Cla",
                "Clb",
                "Cma",
                "Cmb",
                "Cna",
                "Cnb",
                "CLp",
                "CLq",
                "CLr",
                "CYp",
                "CYq",
                "CYr",
                "Clp",
                "Clq",
                "Clr",
                "Cmp",
                "Cmq",
                "Cmr",
                "Cnp",
                "Cnq",
                "Cnr",
                "Xnp",
                "Clb Cnr / Clr Cnb"
            ]

            if len(values) != 57 and len(values) != 56: # Sometimes the spiral mode term is inexplicably not displayed by AVL
                raise RuntimeError(
                    "AVL could not run for some reason!\n"
                    "Investigate by turning on the `verbose` flag and looking at the output.\n"
                    "(Common culprit: angular rates too high.)"
                )

            res = {
                k: v
                for k, v in zip(
                    keys, values
                )
            }

            ##### Add a few more outputs for ease of use
            res["p"] = res["pb/2V"] * (2 * self.op_point.velocity / self.airplane.b_ref)
            res["q"] = res["qc/2V"] * (2 * self.op_point.velocity / self.airplane.c_ref)
            res["r"] = res["rb/2V"] * (2 * self.op_point.velocity / self.airplane.b_ref)

            return res


if __name__ == '__main__':

    ### Import Vanilla Airplane
    import aerosandbox as asb

    from pathlib import Path

    geometry_folder = Path(asb.__file__).parent.parent / "tutorial" / "04 - Geometry" / "example_geometry"

    import sys

    sys.path.insert(0, str(geometry_folder))

    from vanilla import airplane as vanilla

    ### Do the AVL run
    avl = AVL(
        airplane=vanilla,
        op_point=OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=1,
            alpha=0.433476,
            beta=0,
            p=0,
            q=0,
            r=0,
        ),
    )

    res = avl.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
