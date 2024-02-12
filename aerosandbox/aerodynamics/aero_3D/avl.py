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
            component=None,  # This is an int
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
            cl_alpha_factor=None,  # This is a float
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
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 avl_command: str = "avl",
                 verbose: bool = False,
                 timeout: Union[float, int, None] = 5,
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

                * If AVL is not on your system PATH, then you should provide a filepath to the AVL executable.

                Note that AVL is not on your PATH by default. To tell if AVL is on your system PATH, open up a
                terminal and type "avl".

                    * If the AVL menu appears, it's on your PATH.

                    * If you get something like "'avl' is not recognized as an internal or external command..." or
                    "Command 'avl' not found, did you mean...", then it is not on your PATH and you'll need to
                    specify the location of your AVL executable as a string.

                To add AVL to your path, modify your system's environment variables. (Google how to do this for your OS.)

            verbose: Controls whether or not AVL output is printed to command line.

            timeout: Controls how long any individual AVL run is allowed to run before the
            process is killed. Given in units of seconds. To disable timeout, set this to None.

            working_directory: Controls which working directory is used for the AVL input and output files. By
            default, this is set to a TemporaryDirectory that is deleted after the run. However, you can set it to
            somewhere local for debugging purposes.
        """
        super().__init__()

        ### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.avl_command = avl_command
        self.verbose = verbose
        self.timeout = timeout
        self.working_directory = working_directory
        self.ground_effect = ground_effect
        self.ground_effect_height = ground_effect_height

    def __repr__(self):
        return self.__class__.__name__ + "(\n\t" + "\n\t".join([
            f"airplane={self.airplane}",
            f"op_point={self.op_point}",
            f"xyz_ref={self.xyz_ref}",
        ]) + "\n)"

    def open_interactive(self) -> None:
        """
        Opens a new terminal window and runs AVL interactively. This is useful for detailed analysis or debugging.

        Returns: None
        """
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            ### Handle the airplane file
            airplane_file = "airplane.avl"
            self.write_avl(directory / airplane_file)

            ### Open up AVL
            import sys, os
            if sys.platform == "win32":
                # Run AVL
                print("Running AVL interactively in a new window, quit it to continue...")

                command = f'cmd /k "{self.avl_command} {airplane_file}"'

                process = subprocess.Popen(
                    command,
                    cwd=directory,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
                process.wait()

            else:
                raise NotImplementedError(
                    "Ability to auto-launch interactive AVL sessions isn't yet implemented for non-Windows OSes."
                )

    def run(self,
            run_command: str = None,
            ) -> Dict[str, float]:
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

            keystrokes = "\n".join(keystroke_file_contents)

            command = f'{self.avl_command} {airplane_file}'

            ### Execute
            try:
                proc = subprocess.Popen(
                    command,
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
                    input=keystrokes,
                    timeout=self.timeout
                )
                return_code = proc.poll()

            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()

                warnings.warn(
                    "AVL run timed out!\n"
                    "If this was not expected, try increasing the `timeout` parameter\n"
                    "when you create this AeroSandbox AVL instance.",
                    stacklevel=2
                )

            ##### Parse the output file
            # Read the file
            try:
                with open(directory / output_filename, "r") as f:
                    output_data = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "It appears AVL didn't produce an output file, probably because it crashed.\n"
                    "To troubleshoot, try some combination of the following:\n"
                    "\t - In the AVL constructor, verify that either AVL is on PATH or that the `avl_command` parameter is set.\n"
                    "\t - In the AVL constructor, run with `verbose=True`.\n"
                    "\t - In the AVL constructor, set the `working_directory` parameter to a known folder to see the AVL input and output files.\n"
                    "\t - In the AVL constructor, set the `timeout` parameter to a large number to see if AVL is just taking a long time to run.\n"
                    "\t - On Windows, use `avl.open_interactive()` to run AVL interactively in a new window.\n"
                )

            res = self.parse_unformatted_data_output(output_data, data_identifier=" =", overwrite=False)

            ##### Clean up results
            for key_to_lowerize in ["Alpha", "Beta", "Mach"]:
                res[key_to_lowerize.lower()] = res.pop(key_to_lowerize)

            for key in list(res.keys()):
                if "tot" in key:
                    res[key.replace("tot", "")] = res.pop(key)

            ##### Add in missing useful results
            q = self.op_point.dynamic_pressure()
            S = self.airplane.s_ref
            b = self.airplane.b_ref
            c = self.airplane.c_ref

            res["p"] = res["pb/2V"] * (2 * self.op_point.velocity / b)
            res["q"] = res["qc/2V"] * (2 * self.op_point.velocity / c)
            res["r"] = res["rb/2V"] * (2 * self.op_point.velocity / b)
            res["L"] = q * S * res["CL"]
            res["Y"] = q * S * res["CY"]
            res["D"] = q * S * res["CD"]
            res["l_b"] = q * S * b * res["Cl"]
            res["m_b"] = q * S * c * res["Cm"]
            res["n_b"] = q * S * b * res["Cn"]
            try:
                res["Clb Cnr / Clr Cnb"] = res["Clb"] * res["Cnr"] / (res["Clr"] * res["Cnb"])
            except ZeroDivisionError:
                res["Clb Cnr / Clr Cnb"] = np.nan

            res["F_w"] = [
                -res["D"], res["Y"], -res["L"]
            ]
            res["F_b"] = self.op_point.convert_axes(*res["F_w"], from_axes="wind", to_axes="body")
            res["F_g"] = self.op_point.convert_axes(*res["F_b"], from_axes="body", to_axes="geometry")
            res["M_b"] = [
                res["l_b"], res["m_b"], res["n_b"]
            ]
            res["M_g"] = self.op_point.convert_axes(*res["M_b"], from_axes="body", to_axes="geometry")
            res["M_w"] = self.op_point.convert_axes(*res["M_b"], from_axes="body", to_axes="wind")

            return res

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

        # Direct p, q, r to be in body axes, to match ASB convention
        run_file_contents += [
            "o",
            "r",
            "",
        ]

        # Set parameters
        run_file_contents += [
            "m",
            f"mn {float(self.op_point.mach())}",
            f"v {float(self.op_point.velocity)}",
            f"d {float(self.op_point.atmosphere.density())}",
            "g 9.81",
            ""
        ]

        # Set analysis state
        p_bar = self.op_point.p * self.airplane.b_ref / (2 * self.op_point.velocity)
        q_bar = self.op_point.q * self.airplane.c_ref / (2 * self.op_point.velocity)
        r_bar = self.op_point.r * self.airplane.b_ref / (2 * self.op_point.velocity)

        run_file_contents += [
            f"a a {float(self.op_point.alpha)}",
            f"b b {float(self.op_point.beta)}",
            f"r r {float(p_bar)}",
            f"p p {float(q_bar)}",
            f"y y {float(r_bar)}"
        ]

        # Set control surface deflections
        run_file_contents += [
            f"d1 d1 1"
        ]

        return run_file_contents

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

        airplane = self.airplane

        avl_file = ""

        airplane_options = self.get_options(airplane)

        avl_file += clean(f"""\
        {airplane.name}
        #Mach
        0        ! AeroSandbox note: This is overwritten later to match the current OperatingPoint Mach during the AVL run.
        #IYsym   IZsym   Zsym
         0       {1 if self.ground_effect else 0}   {self.ground_effect_height}
        #Sref    Cref    Bref
        {airplane.s_ref} {airplane.c_ref} {airplane.b_ref}
        #Xref    Yref    Zref
        {self.xyz_ref[0]} {self.xyz_ref[1]} {self.xyz_ref[2]}
        # CDp
        {airplane_options["profile_drag_coefficient"]}
        """)

        control_surface_counter = 0
        airfoil_counter = 0

        for wing in airplane.wings:

            wing_options = self.get_options(wing)

            spacing_line = f"{wing_options['chordwise_resolution']}   {self.AVL_spacing_parameters[wing_options['chordwise_spacing']]}"
            if wing_options["wing_level_spanwise_spacing"]:
                spacing_line += f"   {wing_options['spanwise_resolution']}   {self.AVL_spacing_parameters[wing_options['spanwise_spacing']]}"

            avl_file += clean(f"""\
            #{"=" * 79}
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
                    sign_dup = 1 if surf.symmetric else -1

                    command = clean(f"""\
                        CONTROL
                        #name, gain, Xhinge, XYZhvec, SgnDup
                        {surf.name} 1 {xhinge:.8g} 0 0 0 {sign_dup}
                        """)

                    control_surface_commands[i].append(command)
                    control_surface_commands[i + 1].append(command)

            ### Write the commands for each wing section
            for i, xsec in enumerate(wing.xsecs):

                xsec_options = self.get_options(xsec)

                xsec_def_line = f"{xsec.xyz_le[0]:.8g} {xsec.xyz_le[1]:.8g} {xsec.xyz_le[2]:.8g} {xsec.chord:.8g} {xsec.twist:.8g}"
                if not wing_options["wing_level_spanwise_spacing"]:
                    xsec_def_line += f"   {xsec_options['spanwise_resolution']}   {self.AVL_spacing_parameters[xsec_options['spanwise_spacing']]}"



                if xsec_options["cl_alpha_factor"] is None:
                    claf_line = f"{1 + 0.77 * xsec.airfoil.max_thickness()}  # Computed using rule from avl_doc.txt"
                else:
                    claf_line = f"{xsec_options['cl_alpha_factor']}"

                af_filepath = Path(str(filepath) + f".af{airfoil_counter}")
                airfoil_counter += 1
                xsec.airfoil.repanel(50).write_dat(filepath=af_filepath, include_name=True)

                avl_file += clean(f"""\
                #{"-" * 50}
                SECTION
                #Xle    Yle    Zle     Chord   Ainc  [Nspanwise   Sspace]
                {xsec_def_line}
                
                AFIL
                {af_filepath}
                
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
            
            TRANSLATE
            0 {np.mean([x.xyz_c[1] for x in fuse.xsecs]):.8g} 0
            
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
                        f"{xyz_c[0]:.8g} {xyz_c[2] + r:.8g}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in fuselage.xsecs][::-1],
                [xsec.equivalent_radius(preserve="area") for xsec in fuselage.xsecs][::-1]
            )
                    ] + [
                        f"{xyz_c[0]:.8g} {xyz_c[2] - r:.8g}"
                        for xyz_c, r in zip(
                [xsec.xyz_c for xsec in fuselage.xsecs][1:],
                [xsec.equivalent_radius(preserve="area") for xsec in fuselage.xsecs][1:]
            )
                    ]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

    @staticmethod
    def parse_unformatted_data_output(
            s: str,
            data_identifier: str = " = ",
            cast_outputs_to_float: bool = True,
            overwrite: bool = None
    ) -> Dict[str, float]:
        """
        Parses a (multiline) string of unformatted data into a nice and tidy dictionary.

        The expected input string looks like what you might get as an output from AVL (or many other Drela codes),
        which may list data in ragged order.

        An example input `s` that you might want to parse could look like the following:

        ```
         Standard axis orientation,  X fwd, Z down

         Run case:  -unnamed-

          Alpha =   0.43348     pb/2V =  -0.00000     p'b/2V =  -0.00000
          Beta  =   0.00000     qc/2V =   0.00000
          Mach  =     0.003     rb/2V =  -0.00000     r'b/2V =  -0.00000

          CXtot =  -0.02147     Cltot =   0.00000     Cl'tot =   0.00000
          CYtot =   0.00000     Cmtot =   0.28149
          CZtot =  -1.01474     Cntot =  -0.00000     Cn'tot =  -0.00000

          CLtot =   1.01454
          CDtot =   0.02915
          CDvis =   0.00000     CDind = 0.0291513
          CLff  =   1.00050     CDff  = 0.0297201    | Trefftz
          CYff  =   0.00000         e =    0.9649    | Plane
        ```

        Here, this function will go through this string and extract each key-value pair, as denoted by the data
        identifier (by default, " = "). It will pull the next whole word without spaces to the left as the key,
        and it will pull the next whole word without spaces to the right as the value. Together, these will be
        returned as a Dict.

        So, the output for the input above would be:
        {
            'Alpha' : 0.43348,
            'pb/2V' : -0.00000,
            'p'b/2V' : -0.00000,
            'Beta' : 0.00000,
            # and so on...
        }

        Args:

            s: The input string to identify. Can be multiline.

            data_identifier: The triggering substring for a new key-value pair. By default, it's " = ",
            which is convention in many output files from Mark Drela's codes. Be careful if you decide to change this
            to "=", as you could pick up on heading separators ('=======') in Markdown-like files.

            cast_outputs_to_float: If this boolean flag is set true, the values of the key-value pairs are cast to
            floating-point numbers before returning (as opposed to the default type, string). If a value can't be
            cast, a NaN is returned (guaranteeing that you can do floating-point math with the outputs in downstream
            applications.)

            overwrite: Determines the behavior if you find a key that's already in the dictionary.

                * By default, value is None. In this case, an error is raised.

                * If you set it to True, the new value will overwrite the old one. Thus, your dictionary will have
                the last matching value from the string.

                * If you set it to False, the new value will be discarded. Thus, your dictionary will have the first
                matching value from the string.

        Returns: A dictionary of key-value pairs, corresponding to the unformatted data in the input string.

            Keys are strings, values are floats if `cast_outputs_to_float` is True, otherwise also strings.

        """

        items = {}

        index = s.find(data_identifier)

        while index != -1:  # While there are still data identifiers:

            key = ""  # start with a blank key, which we will build up as we read

            i = index - 1  # Starting from the left of the identifier
            while s[i] == " " and i >= 0:
                # First, skip any blanks
                i -= 1
            while s[i] != " " and s[i] != "\n" and i >= 0:
                # Then, read the key in backwards order until you get to a blank or newline
                key = s[i] + key
                i -= 1

            value = ""  # start with a blank value, which we will build up as we read

            i = index + len(data_identifier)  # Starting from the right of the identifier
            while s[i] == " " and i <= len(s):
                # First, skip any blanks
                i += 1
            while s[i] != " " and s[i] != "\n" and i <= len(s):
                # Then, read the key in forward order until you get to a blank or newline
                value += s[i]
                i += 1

            if cast_outputs_to_float:
                try:  # Try to convert the value into a float. If you can't, return a NaN
                    value = float(value)
                except Exception:
                    value = np.nan

            if key in items.keys():  # If you already have this key
                if overwrite is None:  # If the `overwrite` parameter wasn't explicitly defined True/False, raise an error
                    raise ValueError(
                        f"Key \"{key}\" is being overwritten, and no behavior has been specified here (Default behavior is to error).\n"
                        f"Check that the output file doesn't have a duplicate here.\n"
                        f"Alternatively, set the `overwrite` parameter of this function to True or False (rather than the default None).",
                    )
                else:
                    if overwrite:
                        items[key] = value  # Assign (and overwrite) the key-value pair to the output we're writing
                    else:
                        pass
            else:
                items[key] = value  # Assign the key-value pair to the output we're writing

            s = s[index + len(data_identifier):]  # Trim the string by starting to read from the next point.
            index = s.find(data_identifier)

        return items


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
        working_directory=str(Path.home() / "Downloads" / "test"),
        verbose=True
    )

    res = avl.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
