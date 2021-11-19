from aerosandbox.common import ExplicitAnalysis
import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from aerosandbox.geometry import Airplane, Wing, WingXSec, Fuselage
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

    options_for_analysis_defaults = {
            "ground_plane": False,
            "ground_plane_height": 0.0,
            "parasitic_drag_coefficient": 0.0,
            "component": None,
            "no_wake": False,
            "no_alpha_beta": False,
            "no_load": False,
            "drag_polar": None,
            "wing_level_spanwise_spacing": True,
            "cl_alpha_factor": None,
            "spanwise_resolution": 12,
            "spanwise_spacing": "cosine",
            "chordwise_resolution": 12,
            "chordwise_spacing": "cosine",
            "fuse_panel_resolution": 24,
            "fuse_panel_spacing": "cosine"
        }

    options_for_analysis_by_object = {
        Airplane: [
            "ground_plane",
            "ground_plane_height",
            "parasitic_drag_coefficient"
        ],
        Wing: [
            "wing_level_spanwise_spacing",
            "spanwise_resolution",
            "spanwise_spacing",
            "chordwise_resolution",
            "chordwise_spacing",
            "component",
            "no_wake",
            "no_alpha_beta",
            "no_load",
            "drag_polar"
        ],
        WingXSec: [
            "spanwise_resolution",
            "spanwise_spacing",
            "cl_alpha_factor",
            "drag_polar",
        ],
        Fuselage: [
            "fuse_panel_resolution",
            "fuse_panel_spacing"
        ]
    }

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
        control_counter = 0
        for wing in self.airplane.wings:
            for xsec in wing.xsecs[:-1]: # there are n - 1 control surfaces for a wing with n xsecs
                control_counter += 1
                control_name = f"d{control_counter}"
                run_file_contents += [
                    f"{control_name} {control_name} {xsec.control_surface_deflection}"
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
            self.write_avl(self.airplane, directory / airplane_file)

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
    
    @staticmethod
    def write_avl(airplane: Airplane,
                  filepath: Union[Path, str] = None,
                  ) -> str:
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
            Cleans up a multi-line string.
            """
            # return dedent(s)
            return "\n".join([line.strip() for line in s.split("\n")])

        string = ""

        options = airplane.get_analysis_specific_options(__class__)
        
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

        num_control_surface = 1
        for wing in airplane.wings:

            options = wing.get_analysis_specific_options(__class__)
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
                
                options = xsec.get_analysis_specific_options(__class__)

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

                # control surface n is defined using xsec i, spanning the section from xsec i to xsec i + 1
                if idx_xsec == 0: # first xsec in wing
                    idx_xsecs_active = [idx_xsec]
                    idx_control_surfaces_active = [num_control_surface]
                    num_control_surface += 1
                elif idx_xsec == len(wing.xsecs) - 1: # last xsec in wing
                    idx_xsecs_active = [idx_xsec - 1]
                    idx_control_surfaces_active = [num_control_surface - 1]
                else:
                    idx_xsecs_active = [idx_xsec - 1, idx_xsec]
                    idx_control_surfaces_active = [num_control_surface - 1, num_control_surface]
                    num_control_surface += 1
                
                for idx_xsec_active, idx_control_surface_active in zip(idx_xsecs_active, idx_control_surfaces_active):
                    xsec_active = wing.xsecs[idx_xsec_active]
                    sign_duplication = 1.0 if xsec_active.control_surface_is_symmetric else -1.0
                    string += clean(f"""
                    CONTROL
                    #Cname Cgain Xhinge HingeVec SgnDup
                    control{idx_control_surface_active} 1.0 {xsec_active.control_surface_hinge_point} 0.0 0.0 0.0 {sign_duplication}
                    """)
        
        filepath = Path(filepath)
        for i, fuse in enumerate(airplane.fuselages):
            fuse_filepath = Path(str(filepath) + f".fuse{i}")
            __class__.write_avl_bfile(
                fuse,
                filepath=fuse_filepath
            )

            options = fuse.get_analysis_specific_options(__class__)

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

        return string

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

            include_name: Should the name be included in the .dat file? (This should be True for use with AVL.)

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
