from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airplane, Wing, WingXSec, Fuselage, ControlSurface
from aerosandbox.performance import OperatingPoint
from typing import Union, List, Dict, Any
import copy
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

        >>> avl = asb.AVL(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> outputs = avl.run()

    """
    default_analysis_specific_options = {
        Airplane: dict(
            profile_drag_coefficient=0
        ),
        Wing    : dict(
            wing_level_spanwise_spacing=True,
            spanwise_resolution=12,
            spanwise_spacing="cosine",
            chordwise_resolution=12,
            chordwise_spacing="cosine",
            component=None,  # type: int
            no_wake=False,
            no_alpha_beta=False,
            no_load=False,
            drag_polar=dict(
                CL1=0,
                CD1=0,
                CL2=0,
                CD2=0,
                CL3=0,
                CD3=0,
            ),
        ),
        WingXSec: dict(
            spanwise_resolution=12,
            spanwise_spacing="cosine",
            cl_alpha_factor=None, # type: float
            drag_polar=dict(
                CL1=0,
                CD1=0,
                CL2=0,
                CD2=0,
                CL3=0,
                CD3=0,
            )
        ),
        Fuselage: dict(
            panel_resolution=24,
            panel_spacing="cosine"
        )
    }

    AVL_spacing_parameters = {
        "uniform": 0,
        "cosine" : 1,
        "sine"   : 2,
        "-sine"  : -2,
        "equal"  : 0,  # "uniform" is preferred
    }

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint = None,
                 avl_command: str = "avl",
                 verbose: bool = False,
                 working_directory: str = None,
                 ground_effect: bool = False,
                 ground_effect_height: float = 0
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
        ### Set defaults
        if op_point is None:
            op_point = OperatingPoint()

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.avl_command = avl_command
        self.verbose = verbose
        self.working_directory = working_directory
        self.ground_effect = ground_effect
        self.ground_effect_height = ground_effect_height

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
            "m",
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

        # Set control surface deflections
        run_file_contents += [
            f"d1 d1 1"
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
            self.write_avl(directory / airplane_file)

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

            ### Iterate through the string to find all the keys and corresponding numeric values, based on where "=" appears.
            keys = []
            values = []
            index = output_data.find("=")

            while index != -1:

                # All keys contain no spaces except for "Clb Cnr / Clr Cnb" (spiral stability parameter)
                potential_keys = output_data[:index].split()
                if potential_keys[-1] == "Cnb" and len(potential_keys) >= 5:
                    potential_key = " ".join(potential_keys[-5:])
                    if potential_key == "Clb Cnr / Clr Cnb":
                        key = potential_key
                    else:
                        key = potential_keys[-1]
                else:
                    key = potential_keys[-1]

                keys.append(key)
                output_data = output_data[index + 1:]
                number = output_data[:12].split("\n")[0]
                try:
                    number = float(number)
                except:
                    number = np.nan
                values.append(number)

                index = output_data.find("=")

            # Make key names consistent with AeroSandbox notation
            keys_lowerize = ["Alpha", "Beta", "Mach"]
            keys = [key.lower() if key in keys_lowerize else key for key in keys]
            keys = [key.replace("tot", "") for key in keys]

            if len(values) < 56:  # Sometimes the spiral mode term is inexplicably not displayed by AVL
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

    def write_avl(self,
                  filepath: Union[Path, str] = None,
                  ) -> None:
        """
        Writes a .avl file corresponding to this airplane to a filepath.

        For use with the AVL vortex-lattice-method aerodynamics analysis tool by Mark Drela at MIT.
        AVL is available here: https://web.mit.edu/drela/Public/web/avl/

        Args:
            filepath: filepath (including the filename and .avl extension) [string]
                If None, this function returns the .avl file as a string.

        Returns: None

        """

        def clean(s):
            """
            Removes leading and trailing whitespace from each line of a multi-line string.
            """
            return "\n".join([line.strip() for line in s.split("\n")])

        # def print_control_surface(control_surface: ControlSurface,
        #                           options_for_analysis: Dict[type, Dict[str, Any]]
        #                           ) -> str:
        #
        #     if control_surface.trailing_edge:
        #         hinge_point = control_surface.hinge_point
        #     else:
        #         hinge_point = -control_surface.hinge_point  # leading edge surfaces are defined with negative hinge points in AVL
        #
        #     if options_for_analysis["duplication_factor"] is None:
        #         duplication_factor = 1.0 if control_surface.symmetric else -1.0
        #     else:
        #         duplication_factor = options_for_analysis["duplication_factor"]
        #
        #     string = clean(f"""
        #     CONTROL
        #     #Cname Cgain Xhinge HingeVec SgnDup
        #     control{control_surface.id} {options_for_analysis['gain']} {hinge_point} {' '.join(str(x) for x in options_for_analysis['hinge_vector'])} {duplication_factor}
        #     """)
        #
        #     return string
        #
        # def print_control_surface_deprecated(xsec: WingXSec,
        #                                      id: int) -> str:
        #
        #     sign_duplication = 1.0 if xsec.control_surface_is_symmetric else -1.0
        #     string = clean(f"""
        #     CONTROL
        #     #Cname Cgain Xhinge HingeVec SgnDup
        #     control{id} 1.0 {xsec.control_surface_hinge_point} 0.0 0.0 0.0 {sign_duplication}
        #     """)
        #
        #     return string

        airplane = self.airplane

        avl_file = ""

        airplane_options = self.get_options(airplane)

        avl_file += clean(f"""\
        {airplane.name}
        #Mach
        0
        #IYsym   IZsym   Zsym
         0       {1 if self.ground_effect else 0}   {self.ground_effect_height}
        #Sref    Cref    Bref
        {airplane.s_ref} {airplane.c_ref} {airplane.b_ref}
        #Xref    Yref    Zref
        {airplane.xyz_ref[0]} {airplane.xyz_ref[1]} {airplane.xyz_ref[2]}
        # CDp
        {airplane_options["profile_drag_coefficient"]}
        """)

        control_surface_counter = 1
        for wing in airplane.wings:

            wing_options = self.get_options(wing)

            spacing_line = f"{wing_options['chordwise_resolution']}   {self.AVL_spacing_parameters[wing_options['chordwise_spacing']]}"
            if wing_options["wing_level_spanwise_spacing"]:
                spacing_line += f"   {wing_options['spanwise_resolution']}   {self.AVL_spacing_parameters[wing_options['spanwise_spacing']]}"

            avl_file += clean(f"""\
            #{"=" * 50}
            SURFACE
            {wing.name}
            #Nchordwise  Cspace  [Nspanwise   Sspace]
            {spacing_line}
            
            """)

            if wing_options["component"] is not None:
                avl_file += clean(f"""\
                COMPONENT
                {wing_options['component']}
                    
                """)

            if wing.symmetric:
                avl_file += clean(f"""\
                YDUPLICATE
                0
                    
                """)

            if wing_options["no_wake"]:
                avl_file += clean(f"""\
                NOWAKE
                
                """)

            if wing_options["no_alpha_beta"]:
                avl_file += clean(f"""\
                NOALBE
                
                """)

            if wing_options["no_load"]:
                avl_file += clean(f"""\
                NOLOAD
                
                """)

            polar = wing_options["drag_polar"]
            avl_file += clean(f"""\
            CDCL
            #CL1  CD1  CL2  CD2  CL3  CD3
            {polar["CL1"]} {polar["CD1"]} {polar["CL2"]} {polar["CD2"]} {polar["CL3"]} {polar["CD3"]}
            
            """)

            ### Build up a buffer of the control surface strings to write to each section
            control_surface_commands: List[List[str]] = [
                []
                for _ in wing.xsecs
            ]
            for i, xsec in enumerate(wing.xsecs[:-1]):
                for surf in xsec.control_surfaces:
                    xhinge = surf.hinge_point if surf.trailing_edge else -surf.hinge_point
                    xyz_hinge_vector = wing._compute_frame_of_section(i)[1]
                    sign_dup = 1 if surf.symmetric else -1

                    command = clean(f"""\
                        CONTROL
                        #name, gain, Xhinge, XYZhvec, SgnDup
                        all_deflections {surf.deflection} {xhinge} {xyz_hinge_vector[0]} {xyz_hinge_vector[1]} {xyz_hinge_vector[2]} {sign_dup}
                        """)

                    control_surface_commands[i].append(command)
                    control_surface_commands[i + 1].append(command)

            ### Write the commands for each wing section
            for i, xsec in enumerate(wing.xsecs):

                xsec_options = self.get_options(xsec)

                xsec_def_line = f"{xsec.xyz_le[0]} {xsec.xyz_le[1]} {xsec.xyz_le[2]} {xsec.chord} {xsec.twist}"
                if not wing_options["wing_level_spanwise_spacing"]:
                    xsec_def_line += f"   {xsec_options['spanwise_resolution']}   {self.AVL_spacing_parameters[xsec_options['spanwise_spacing']]}"

                if xsec_options["cl_alpha_factor"] is None:
                    claf_line = f"{1 + 0.77 * xsec.airfoil.max_thickness()}  # Computed using rule from avl_doc.txt"
                else:
                    claf_line = f"{xsec_options['cl_alpha_factor']}"

                avl_file += clean(f"""\
                #{"-" * 50}
                SECTION
                #Xle    Yle    Zle     Chord   Ainc  [Nspanwise   Sspace]
                {xsec_def_line}
                
                AIRFOIL
                {xsec.airfoil.repanel(50).write_dat(filepath=None, include_name=False)}
                
                CLAF
                {claf_line}
                
                """)

                polar = xsec_options["drag_polar"]
                avl_file += clean(f"""\
                CDCL
                #CL1  CD1  CL2  CD2  CL3  CD3
                {polar["CL1"]} {polar["CD1"]} {polar["CL2"]} {polar["CD2"]} {polar["CL3"]} {polar["CD3"]}
                
                """)

                for control_surface_command in control_surface_commands[i]:
                    avl_file += control_surface_command

        filepath = Path(filepath)
        for i, fuse in enumerate(airplane.fuselages):
            fuse_filepath = Path(str(filepath) + f".fuse{i}")
            self.write_avl_bfile(
                fuselage=fuse,
                filepath=fuse_filepath
            )
            fuse_options = self.get_options(fuse)

            avl_file += clean(f"""\
            #{"=" * 50}
            BODY
            {fuse.name}
            {fuse_options['panel_resolution']} {self.AVL_spacing_parameters[fuse_options['panel_spacing']]}
            
            BFIL
            {fuse_filepath}
            
            """)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(avl_file)

    @staticmethod
    def write_avl_bfile(fuselage,
                        filepath: Union[Path, str] = None,
                        include_name: bool = True,
                        ) -> str:
        """
        Writes an AVL-compatible BFILE corresponding to this fuselage to a filepath.

        For use with the AVL vortex-lattice-method aerodynamics analysis tool by Mark Drela at MIT.
        AVL is available here: https://web.mit.edu/drela/Public/web/avl/

        Args:
            filepath: filepath (including the filename and .avl extension) [string]
                If None, this function returns the would-be file contents as a string.

            include_name: Should the name of the fuselage be included in the .dat file? (This should be True for use with AVL.)

        Returns:

        """
        filepath = Path(filepath)

        contents = []

        if include_name:
            contents += [fuselage.name]

        contents += [
                        f"{xyz_c[0]} {xyz_c[2] + r}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in fuselage.xsecs][::-1],
                [xsec.radius for xsec in fuselage.xsecs][::-1]
            )
                    ] + [
                        f"{xyz_c[0]} {xyz_c[2] - r}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in fuselage.xsecs][1:],
                [xsec.radius for xsec in fuselage.xsecs][1:]
            )
                    ]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string


if __name__ == '__main__':

    ### Import Vanilla Airplane
    import aerosandbox as asb

    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import airplane as vanilla

    vanilla.analysis_specific_options[AVL] = dict(
        profile_drag_coefficient=0.1
    )
    vanilla.wings[0].xsecs[0].control_surfaces.append(
        ControlSurface(
            name="Flap",
            trailing_edge=True,
            hinge_point=0.75,
            symmetric=True,
            deflection=10
        )
    )
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
        # working_directory=str(Path.home() / "Downloads" / "avl_test"),
        # verbose=True
    )

    res = avl.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
