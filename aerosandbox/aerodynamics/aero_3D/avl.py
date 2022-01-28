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
        Airplane      : dict(
            parasite_drag_coefficient=0
        ),
        Wing          : dict(
            wing_level_spanwise_spacing=True,
            spanwise_resolution=12,
            spanwise_spacing="cosine",
            chordwise_resolution=12,
            chordwise_spacing="cosine",
            component=None,
            no_wake=False,
            no_alpha_beta=False,
            no_load=False,
            drag_polar=None,
        ),
        WingXSec      : dict(
            spanwise_resolution=12,
            spanwise_spacing="cosine",
            cl_alpha_factor=1,
            drag_polar=None,
        ),
        ControlSurface: dict(
            gain=1,
            hinge_vector=None,
            duplication_factor=None,
        ),
        Fuselage      : dict(
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

            if len(values) != 57 and len(
                    values) != 56:  # Sometimes the spiral mode term is inexplicably not displayed by AVL
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

    @staticmethod
    def write_avl(airplane: Airplane,
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

        def print_control_surface(control_surface: ControlSurface,
                                  options_for_analysis: Dict[type, Dict[str, Any]]
                                  ) -> str:

            if control_surface.trailing_edge:
                hinge_point = control_surface.hinge_point
            else:
                hinge_point = -control_surface.hinge_point # leading edge surfaces are defined with negative hinge points in AVL

            if options_for_analysis["duplication_factor"] is None:
                duplication_factor = 1.0 if control_surface.symmetric else -1.0
            else:
                duplication_factor = options_for_analysis["duplication_factor"]

            string = clean(f"""
            CONTROL
            #Cname Cgain Xhinge HingeVec SgnDup
            control{control_surface.id} {options_for_analysis['gain']} {hinge_point} {' '.join(str(x) for x in options_for_analysis['hinge_vector'])} {duplication_factor}
            """)

            return string

        def print_control_surface_deprecated(xsec: WingXSec,
                                             id: int) -> str:

            sign_duplication = 1.0 if xsec.control_surface_is_symmetric else -1.0
            string = clean(f"""
            CONTROL
            #Cname Cgain Xhinge HingeVec SgnDup
            control{id} 1.0 {xsec.control_surface_hinge_point} 0.0 0.0 0.0 {sign_duplication}
            """)

            return string

        string = ""

        options = airplane.get_options_for_analysis(__class__)

        z_symmetry = 1 if options["ground_plane"] == True else 0

        string += clean(f"""\
        {airplane.name}
        #Mach
        0
        #IYsym   IZsym   Zsym
         0       {z_symmetry}       {options["ground_plane_height"]}
        #Sref    Cref    Bref
        {airplane.s_ref} {airplane.c_ref} {airplane.b_ref}
        #Xref    Yref    Zref
        {airplane.xyz_ref[0]} {airplane.xyz_ref[1]} {airplane.xyz_ref[2]}
        # CDp
        {options["parasitic_drag_coefficient"]}
        """)

        spacing = {
            "uniform": 0.0,
            "cosine": 1.0,
            "sine": 2.0
        }

        control_surface_counter = 1
        for wing in airplane.wings:

            options = wing.get_options_for_analysis(__class__)
            wing_options = copy.deepcopy(options) # store these for comparison to xsec options

            spacing_line = f"{options['chordwise_resolution']}   {spacing[options['chordwise_spacing']]}"
            if options["wing_level_spanwise_spacing"] == True:
                spacing_line += f"   {options['spanwise_resolution']}   {spacing[options['spanwise_spacing']]}"

            string += clean(f"""\
            #{"=" * 50}
            SURFACE
            {wing.name}
            #Nchordwise  Cspace  [Nspanwise   Sspace]
            {spacing_line}

            """)

            if wing.symmetric == True:
                string += clean(f"""\
                YDUPLICATE
                0

                """)

            if options["no_wake"] == True:
                string += clean(f"""\
                NOWAKE
                
                """)

            if options["no_alpha_beta"] == True:
                string += clean(f"""\
                NOALBE
                
                """)

            if options["no_load"] == True:
                string += clean(f"""\
                NOLOAD
                
                """)

            if options["drag_polar"] is not None:
                drag_polar = options["drag_polar"]
                CL = drag_polar["CL"]
                CD = drag_polar["CD"]
                string += clean(f"""\
                CDCL
                #CL1  CD1  CL2  CD2  CL3  CD3
                {' '.join(str(x) for x in [CL[0], CD[0], CL[1], CD[1], CL[2], CD[2]])}
                
                """)

            for idx_xsec, xsec in enumerate(wing.xsecs):

                options = xsec.get_options_for_analysis(__class__)

                xsec_def_line = f"{xsec.xyz_le[0]} {xsec.xyz_le[1]} {xsec.xyz_le[2]} {xsec.chord} {xsec.twist}"
                if wing_options["wing_level_spanwise_spacing"] == False:
                    xsec_def_line += f"   {options['spanwise_resolution']}   {spacing[options['spanwise_spacing']]}"

                if options["cl_alpha_factor"] is None:
                    claf_line = f"{1 + 0.77 * xsec.airfoil.max_thickness()} # Computed using rule from avl_doc.txt"
                else:
                    claf_line = f"{options['cl_alpha_factor']}"

                string += clean(f"""\
                #{"-" * 50}
                SECTION
                #Xle    Yle    Zle     Chord   Ainc  [Nspanwise   Sspace]
                {xsec_def_line}
                
                AIRFOIL
                {xsec.airfoil.repanel(50).write_dat(filepath=None, include_name=False)}
                
                CLAF
                {claf_line}
                """)

                if options["drag_polar"] is not None:
                    drag_polar = options["drag_polar"]
                    CL = drag_polar["CL"]
                    CD = drag_polar["CD"]
                    string += clean(f"""\
                    CDCL
                    #CL1  CD1  CL2  CD2  CL3  CD3
                    {' '.join(str(x) for x in [CL[0], CD[0], CL[1], CD[1], CL[2], CD[2]])}
                    
                    """)

                # see WingXSec in wing.py for explanation of control surface implementation protocol using control_surfaces list vs. deprecated WingXSec properties
                if idx_xsec > 0: # if this xsec is not the first xsec, get previous xsec's control surface info
                    xsec_prev = wing.xsecs[idx_xsec - 1]
                    if xsec_prev.control_surfaces is not None: # if user specifies control_surfaces as None, then there will be no control surface
                        if xsec_prev.control_surfaces: # if control_surfaces is not an empty list, print control surfaces to file using list of ControlSurface instances
                            for control_surface in xsec_prev.control_surfaces:
                                options = control_surface.get_options_for_analysis(__class__)
                                string += print_control_surface(control_surface, options)
                        else: # if control_surfaces is an empty list (default), print control surfaces to file using deprecated WingXSec properties
                            string += print_control_surface_deprecated(xsec_prev, control_surface_counter - 1)

                if idx_xsec < len(wing.xsecs) - 1: # if this xsec is not the last xsec, get this xsec's control surface info
                    if xsec.control_surfaces is not None:
                        if xsec.control_surfaces:
                            for control_surface in xsec.control_surfaces:
                                control_surface.id = control_surface_counter
                                control_surface_counter += 1
                                options = control_surface.get_options_for_analysis(__class__)
                                string += print_control_surface(control_surface, options)
                        else:
                            string += print_control_surface_deprecated(xsec, control_surface_counter)
                            control_surface_counter += 1

        filepath = Path(filepath)
        for i, fuse in enumerate(airplane.fuselages):
            fuse_filepath = Path(str(filepath) + f".fuse{i}")
            __class__.write_avl_bfile(
                fuse,
                filepath=fuse_filepath
            )

            options = fuse.get_options_for_analysis(__class__)

            string += clean(f"""\
            #{"=" * 50}
            BODY
            {fuse.name}
            {options['fuse_panel_resolution']} {spacing[options['fuse_panel_spacing']]}
            
            BFIL
            {fuse_filepath}
            
            """)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

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
        working_directory=str(Path.home() / "Downloads" / "avl_test"),
        verbose=True
    )

    res = avl.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
