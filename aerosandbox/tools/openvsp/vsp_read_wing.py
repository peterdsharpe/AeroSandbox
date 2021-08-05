import aerosandbox
from aerosandbox.geometry.airfoil import Airfoil 
from aerosandbox.geometry import Wing
from aerosandbox.geometry import WingXSec
import openvsp as vsp
#import aerosandbox.numpy as np
import numpy as np
import string
from ctypes import *

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

    Outputs:
    Writes aerosandbox wing object

    Properties Used:
    N/A
    """  
    print("Converting wing: " + wing_id)
    x_rot = vsp.GetParmVal(wing_id,'X_Rotation','XForm')
    
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
    total_proj_span = vsp.GetParmVal(wing_id, 'TotalProjectedSpan', 'WingGeom')
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)   # This is how VSP stores surfaces.
    segment_num = vsp.GetNumXSec(xsec_root_id)    # Get number of wing segments (is one more than the VSP GUI shows).
    x_sec = vsp.GetXSec(xsec_root_id, 0)
    chord_parm = vsp.GetXSecParm(x_sec,'Root_Chord')
    total_chord = vsp.GetParmVal(chord_parm) 
    x_rot = vsp.GetParmVal( wing_id,'X_Rotation','XForm')
    vertical = False
    if x_rot > 70:
        vertical = True
    
    # -------------
    # Wing segments
    # -------------
    start = 0
    xsecs = []
    # Convert VSP XSecs to aerosandbox segments.
    for increment in range(start, segment_num):    
        xsec_next = getWingXsec(wing_id, total_proj_span, symmetric, segment_num, increment, vertical)
        xsecs.append(xsec_next)
    return Wing(tag, xyz_le, xsecs, symmetric)

def getWingXsec(wing_id, total_proj_span, symmetric, segment_num, increment, vertical):
    print("   Processing xsec: " + str(increment) + " for wing: " + wing_id)
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    xsec = vsp.GetXSec(xsec_root_id, increment)
    point = vsp.ComputeXSecPnt(xsec, 0.0)    # get xsec point at leading edge
    p = (c_double * 3).from_address(int(np.ctypeslib.as_array(point.data())))
    if vertical:
        xyz_le = np.array([0, 0, 0])
        xyz_le[0] = p[0]
        xyz_le[1] = p[2]
        xyz_le[2] = p[1]
    else:
        xyz_le =  np.array(list(p))
    print("      xyz_le: " + str(xyz_le))
    chord_root = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(increment))
    print("      Root_Chord_Xsec: " + str(chord_root))
    chord = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(increment))
    print("      Tip_Chord_Xsec: " + str(chord))
    twist = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(increment))
    print("      Twist: " + str(twist))
    airfoil = getXsecAirfoil(wing_id, xsec, increment)
    return WingXSec(xyz_le, chord, twist, airfoil=airfoil)

def getXsecAirfoil(wing_id, xsec_id, xsec_num):
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    total_xsec = vsp.GetNumXSec(xsec_root_id)
    print("   Processing airfoil: " + str(xsec_num) + " for wing: " + wing_id)
    thick_cord = vsp.GetParmVal(wing_id, 'ThickChord', 'XSecCurve_' + str(xsec_num))
    if vsp.GetXSecShape(xsec_id) == vsp.XS_FOUR_SERIES:     # XSec shape: NACA 4-series
         camber = vsp.GetParmVal(wing_id, 'Camber', 'XSecCurve_' + str(xsec_num)) 
         if camber == 0.:
             camber_loc = 0.
         else:
             camber_loc = vsp.GetParmVal(wing_id, 'CamberLoc', 'XSecCurve_' + str(xsec_num))
         camber_round               = int(np.around(camber*100))
         camber_loc_round           = int(np.around(camber_loc*10)) 
         thick_cord_round           = int(np.around(thick_cord*100))
         tag                = 'naca' + str(camber_round) + str(camber_loc_round) + str(thick_cord_round)    
         print("   Airfoil is XS_FOUR_SERIES: " + tag)
         return Airfoil(name=tag)
    elif vsp.GetXSecShape(xsec_id) == vsp.XS_SIX_SERIES:     # XSec shape: NACA 6-series
         thick_cord_round = int(np.around(thick_cord*100))
         a_value          = vsp.GetParmVal(wing_id, 'A', 'XSecCurve_' + str(xsec_num))
         ideal_CL         = int(np.around(vsp.GetParmVal(wing_id, 'IdealCl', 'XSecCurve_' + str(xsec_num))*10))
         series_vsp       = int(vsp.GetParmVal(wing_id, 'Series', 'XSecCurve_' + str(xsec_num)))
         series_dict      = {0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'} # VSP series values.
         series           = series_dict[series_vsp]
         tag      = 'naca' + series + str(ideal_CL) + str(thick_cord_round)            
         print("   Airfoil is XS_SIX_SERIES: " + tag)
         return Airfoil(name=tag)
    elif vsp.GetXSecShape(xsec_id) == vsp.XS_FILE_AIRFOIL:    # XSec shape: 12 is type AF_FILE
         print("   Airfoil is XS_FILE_AIRFOIL")
         name=str(wing_id) + '_airfoil_XSec_' + str(xsec_num)
         vsp.WriteSeligAirfoil(name +'.dat', wing_id, float(xsec_num/total_xsec))
         return Airfoil(name=name, coordinates=name +'.dat')
    else:
         print("   Error:  Could not determine airfoil")

def getXSecSpans(wing_id):
    xsec_spans = []
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    segment_num = vsp.GetNumXSec(xsec_root_id)
    for increment in range(0, segment_num):
       xsec_id = str(vsp.GetXSec(xsec_root_id, increment))
       segment_span = vsp.GetParmVal(wing_id, 'Span', 'XSec_' + str(i))
       xsec_spans.append(segment_span)
    return xsec_spans

def getXSecDihedrals(wing_id):
    xsec_dihedral = []
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    segment_num = vsp.GetNumXSec(xsec_root_id)
    for increment in range(0, segment_num):
       xsec_id = str(vsp.GetXSec(xsec_root_id, increment))
       xsec_dihedral.append()
    return xsec_dihedral

def getXSecSweepsQuarterChord(wing_id):
    xsec_sweeps_quarter_chord = []
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    segment_num = vsp.GetNumXSec(xsec_root_id)
    for increment in range(0, segment_num):
       xsec_id = str(vsp.GetXSec(xsec_root_id, increment))
       sweep = vsp.GetParmVal(wing_id, 'Sweep', 'XSec_' + str(i)) * Units.deg
       sweep_loc = vsp.GetParmVal(wing_id, 'Sweep_Location', 'XSec_' + str(i))
       aspect_ration = 2*vsp.GetParmVal(wing_id, 'Aspect', 'XSec_' + str(i))
       taper = vsp.GetParmVal(wing_id, 'Taper', 'XSec_' + str(i))
       c_4_sweep = convert_sweep(sweep,sweep_loc,0.25,aspect_ratio,taper)
       xsec_sweeps_quarter_chord.append(c_4_sweep)
    return xsec_sweeps_quarter_chord

def convert_sweep(sweep,sweep_loc,new_sweep_loc,aspect_ratio,taper):
    sweep_LE = np.arctan(np.tan(sweep)+4*sweep_loc*
                              (1-taper)/(aspect_ratio*(1+taper))) 
    new_sweep = np.arctan(np.tan(sweep_LE)-4*new_sweep_loc*
                          (1-taper)/(aspect_ratio*(1+taper))) 
    return new_sweep
