## @ingroup Input_Output-OpenVSP
# vsp_read_wing.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero
#           May 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import aerosandbox
from aerosandbox.geometry.airfoil import Airfoil 
from aerosandbox.geometry import Wing
import openvsp as vsp
import aerosandbox.numpy as np
import string

# This enforces lowercase names
chars = string.punctuation + string.whitespace
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                         '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  vsp read wing
# ----------------------------------------------------------------------

## @ingroup Input_Output-OpenVSP
def vsp_read_wing(wing_id, write_airfoil_file=True):     
    """This reads an OpenVSP wing vehicle geometry and writes it into a aerosandbox wing format.

    Assumptions:
    1. OpenVSP wing is divided into segments ("XSecs" in VSP).
    2. Written for OpenVSP 3.21.1

    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for wing.
    2. units_type set to 'SI' (default) or 'Imperial'.

    Outputs:
    Writes aerosandbox wing object

    Properties Used:
    N/A
    """  
    
    # Apply a tag to the wing
    if vsp.GetGeomName(wing_id):
        tag = vsp.GetGeomName(wing_id)
        tag = tag.translate(t_table)
        tag = tag
    else: 
        tag = 'winggeom'
    
    # Wing origin
        xyz_le = np.array([0, 0, 0])
        xyz_le[0] = vsp.GetParmVal(wing_id, 'X_Location', 'XForm')
        xyz_le[1] = vsp.GetParmVal(wing_id, 'Y_Location', 'XForm')
        xyz_le[2] = vsp.GetParmVal(wing_id, 'Z_Location', 'XForm')
    
    # Wing Symmetry
    sym_planar = vsp.GetParmVal(wing_id, 'Sym_Planar_Flag', 'Sym')
    sym_origin = vsp.GetParmVal(wing_id, 'Sym_Ancestor', 'Sym')

    # Check for symmetry
    if sym_planar == 2. and sym_origin == 1.: #origin at wing, not vehicle
        symmetric = True    
    else:
        symmetric = False
        

    
    #More top level parameters
    total_proj_span      = vsp.GetParmVal(wing_id, 'TotalProjectedSpan', 'WingGeom')
    aspect_ratio    = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
    area = vsp.GetParmVal(wing_id, 'TotalArea', 'WingGeom')
 
    # Check if this is a single segment wing
    xsec_surf_id      = vsp.GetXSecSurf(wing_id, 0)   # This is how VSP stores surfaces.
    x_sec_1           = vsp.GetXSec(xsec_surf_id, 1)
    x_sec_1_span_parm = vsp.GetXSecParm(x_sec_1,'Span')
    x_sec_1_span      = vsp.GetParmVal(x_sec_1_span_parm)*(1+symmetric)  # symmetric is 1 if True
    
    if x_sec_1_span == total_proj_span:
        single_seg = True
    else:
        single_seg = False

    segment_num       = vsp.GetNumXSec(xsec_surf_id)    # Get number of wing segments (is one more than the VSP GUI shows).
    x_sec             = vsp.GetXSec(xsec_surf_id, 0)
    chord_parm        = vsp.GetXSecParm(x_sec,'Root_Chord')
    total_chord      = vsp.GetParmVal(chord_parm) 
    span_sum         = 0.                # Non-projected.
    proj_span_sum    = 0.                # Projected.
    segment_spans    = [None] * (segment_num)     # Non-projected.
    segment_dihedral = [None] * (segment_num)
    segment_sweeps_quarter_chord = [None] * (segment_num)
    
    # -------------
    # Wing segments
    # -------------
    start = 0
    root_chord = total_chord 
    if single_seg == False:
        # Convert VSP XSecs to aerosandbox segments.
        for increment in range(start, segment_num+1):    
            xsec = getWingXsec(wing_id, root_chord, total_proj_span, proj_span_sum, symmetric, segment_num, increment)
    else:
        # Single segment

        # Get ID's
        x_sec_1_dih_parm       = vsp.GetXSecParm(x_sec_1,'Dihedral')
        x_sec_1_sweep_parm     = vsp.GetXSecParm(x_sec_1,'Sweep')
        x_sec_1_sweep_loc_parm = vsp.GetXSecParm(x_sec_1,'Sweep_Location')
        x_sec_1_taper_parm     = vsp.GetXSecParm(x_sec_1,'Taper')
        x_sec_1_rc_parm        = vsp.GetXSecParm(x_sec_1,'Root_Chord')
        x_sec_1_tc_parm        = vsp.GetXSecParm(x_sec_1,'Tip_Chord')

        # Calcs
        sweep     = vsp.GetParmVal(x_sec_1_sweep_parm)
        sweep_loc = vsp.GetParmVal(x_sec_1_sweep_loc_parm)
        taper     = vsp.GetParmVal(x_sec_1_taper_parm)
        c_4_sweep = convert_sweep(sweep,sweep_loc,0.25,aspect_ratio,taper)

        # Pull and pack
        sweeps.quarter_chord  = c_4_sweep
        taper                 = taper
        dihedral              = vsp.GetParmVal(x_sec_1_dih_parm) * x_rot
        chords.root           = vsp.GetParmVal(x_sec_1_rc_parm)
        chords.tip            = vsp.GetParmVal(x_sec_1_tc_parm)
        chords.mean_geometric = area / total_proj_span

    # Wing dihedral
    proj_span_sum_alt = 0.
    span_sum_alt      = 0.
    sweeps_sum        = 0.

    for increment in range(start, segment_num):
        span_sum_alt += segment_spans[increment]
        proj_span_sum_alt += segment_spans[increment] * np.cos(segment_dihedral[increment])  # Use projected span to find total wing dihedral.
        sweeps_sum += segment_spans[increment] * np.tan(segment_sweeps_quarter_chord[increment])

    wing.dihedral              = np.arccos(proj_span_sum_alt / span_sum_alt)
    wing.sweeps.quarter_chord  = -np.arctan(sweeps_sum / span_sum_alt)  # Minus sign makes it positive sweep.

    # Add a tip segment, all values are zero except the tip chord
    tc = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(segment_num-1))

    # Chords
    chords_root              = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_0')
    chords_tip               = tc
    chords_mean_geometric    = wing.areas.reference / wing.spans.projected

    # Twists
    twists_root      = vsp.GetParmVal(wing_id, 'Twist', 'XSec_0')
    twists_tip       = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(segment_num-1))
    return wing

def getWingXsec(wing_id, root_chord, total_proj_span, proj_span_sum, symmetric, segment_num, increment):
    # XSec airfoil
    tag                   = 'Section_' + str(increment)
    thick_cord                    = vsp.GetParmVal(wing_id, 'ThickChord', 'XSecCurve_' + str(increment))
    segment_root_chord    = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(increment))
    root_chord_percent    = segment_root_chord / root_chord
    percent_span_location = proj_span_sum / (total_proj_span/(1+symmetric))
    twist                 = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(increment))
            
    if increment < segment_num:      # This excludes the tip xsec.
        sweep     = vsp.GetParmVal(wing_id, 'Sweep', 'XSec_' + str(increment))
        sweep_loc = vsp.GetParmVal(wing_id, 'Sweep_Location', 'XSec_' + str(increment))
        AR        = 2*vsp.GetParmVal(wing_id, 'Aspect', 'XSec_' + str(increment))
        taper     = vsp.GetParmVal(wing_id, 'Taper', 'XSec_' + str(increment))
        segment_sweeps_quarter_chord[incremement] = convert_sweep(sweep,sweep_loc,0.25,AR,taper)
        # Used for dihedral computation, below.
        segment_dihedral        = vsp.GetParmVal(wing_id, 'Dihedral', 'XSec_' + str(increment)) * x_rot
        segment_spans           = vsp.GetParmVal(wing_id, 'Span', 'XSec_' + str(increment))
        proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])    
        span_sum      += segment_spans[i]
    else:
        root_chord_percent    = (vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(increment-1))) /root_chord
        

    xsec_id = str(vsp.GetXSec(xsec_surf_id, increment))
    if vsp.GetXSecShape(xsec_id) == vsp.XS_FOUR_SERIES:     # XSec shape: NACA 4-series
         camber = vsp.GetParmVal(wing_id, 'Camber', 'XSecCurve_' + str(jj)) 
         if camber == 0.:
             camber_loc = 0.
         else:
             camber_loc = vsp.GetParmVal(wing_id, 'CamberLoc', 'XSecCurve_' + str(jj))
         airfoil.thickness_to_chord = thick_cord
         camber_round               = int(np.around(camber*100))
         camber_loc_round           = int(np.around(camber_loc*10)) 
         thick_cord_round           = int(np.around(thick_cord*100))
         airfoil.tag                = 'NACA ' + str(camber_round) + str(camber_loc_round) + str(thick_cord_round)    
    elif vsp.GetXSecShape(xsec_id) == vsp.XS_SIX_SERIES:     # XSec shape: NACA 6-series
         thick_cord_round = int(np.around(thick_cord*100))
         a_value          = vsp.GetParmVal(wing_id, 'A', 'XSecCurve_' + str(jj))
         ideal_CL         = int(np.around(vsp.GetParmVal(wing_id, 'IdealCl', 'XSecCurve_' + str(jj))*10))
         series_vsp       = int(vsp.GetParmVal(wing_id, 'Series', 'XSecCurve_' + str(jj)))
         series_dict      = {0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'} # VSP series values.
         series           = series_dict[series_vsp]
         airfoil.tag      = 'NACA ' + series + str(ideal_CL) + str(thick_cord_round) + ' a=' + str(np.around(a_value,1))            
    elif vsp.GetXSecShape(xsec_id) == vsp.XS_FILE_AIRFOIL:    # XSec shape: 12 is type AF_FILE
         airfoil.thickness_to_chord = thick_cord
                
    return xsec
                
            

def convert_sweep(sweep,sweep_loc,new_sweep_loc,AR,taper):
    sweep_LE = np.arctan(np.tan(sweep)+4*sweep_loc*
                              (1-taper)/(AR*(1+taper))) 
    new_sweep = np.arctan(np.tan(sweep_LE)-4*new_sweep_loc*
                          (1-taper)/(AR*(1+taper))) 
    return new_sweep
