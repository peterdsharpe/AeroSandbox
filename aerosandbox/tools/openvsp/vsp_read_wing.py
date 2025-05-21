import math
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
    2. Written for OpenVSP 3.24

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for wing.

    Outputs:
    returns aerosandbox wing object
    """  
    print("Converting wing: " + wing_id)
    # Apply a tag to the wing
    if vsp.GetGeomName(wing_id):
        tag = vsp.GetGeomName(wing_id)
        tag = tag.translate(t_table)
        tag = tag
    else: 
        tag = 'winggeom'

    # Wing rotation
    xyz_rot = np.array([0, 0, 0])
    xyz_rot[0] = vsp.GetParmVal(wing_id,'X_Rotation','XForm')
    xyz_rot[1] = vsp.GetParmVal(wing_id,'Y_Rotation','XForm')
    xyz_rot[2] = vsp.GetParmVal(wing_id,'Z_Rotation','XForm')
    print("   wing xyz_rot: " + str(xyz_rot))
    
    # Wing origin
    xyz_le = np.array([0, 0, 0])
    xyz_le[0] = vsp.GetParmVal(wing_id, 'X_Location', 'XForm')
    xyz_le[1] = vsp.GetParmVal(wing_id, 'Y_Location', 'XForm')
    xyz_le[2] = vsp.GetParmVal(wing_id, 'Z_Location', 'XForm')
    print("   wing xyz_le: " + str(xyz_le))
    
    # Wing Symmetry
    sym_planar = vsp.GetParmVal(wing_id, 'Sym_Planar_Flag', 'Sym')
    sym_origin = vsp.GetParmVal(wing_id, 'Sym_Ancestor', 'Sym')

    # Check for symmetry
    if sym_planar == 2. and sym_origin == 1.: #origin at wing, not vehicle
        symmetric = True    
    else:
        symmetric = False
        
    #More top level parameters
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)   # This is how VSP stores surfaces.
    segment_num = vsp.GetNumXSec(xsec_root_id)    # Get number of wing segments (is one more than the VSP GUI shows).
    
    # wing segments
    xsecs = []
    # Convert VSP XSecs to aerosandbox segments. 
    for increment in range(0, segment_num):    
        xsec_next = getWingXsec(wing_id, xyz_rot, symmetric, segment_num, increment, "tip")
        xsecs.append(xsec_next)
    return Wing(tag, xyz_le, xsecs, symmetric)

# create a aerosandbox xsec by looking at the openvsp xsec
def getWingXsec(wing_id, xyz_rot, symmetric, segment_num, increment, chord_type):
    print("   Processing xsec: " + str(increment) + " for wing: " + wing_id + " chord type: " + str(chord_type))
    chord_root = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(increment))
    print("      Root_Chord_Xsec: " + str(chord_root))
    chord = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(increment))
    print("      Tip_Chord_Xsec: " + str(chord))
    xsec_root_id = vsp.GetXSecSurf(wing_id, 0)
    xsec = vsp.GetXSec(xsec_root_id, increment)
    if chord_type == "root":
        point = vsp.ComputeXSecPnt(xsec, 0.5)    # get xsec point at leading edge of root chord. (0/1 are trailing edge)
        chord = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(increment))
    else: #chord type is tip
        point = vsp.ComputeXSecPnt(xsec, 0.5)    # get xsec point at leading edge of tip chord
        chord = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(increment))
    p = (c_double * 3).from_address(int(np.ctypeslib.as_array(point.data())))

    # transform point using the wing rotation matrix
    xyz_le = np.array(list(p))
    xyz_le = x_rotation(xyz_le, math.radians(xyz_rot[0]))
    xyz_le = y_rotation(xyz_le, math.radians(xyz_rot[1]))
    xyz_le = z_rotation(xyz_le, math.radians(xyz_rot[2]))
    print("      xyz_le after rotation: " + str(xyz_le))

    twist = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(increment))
    print("      Twist: " + str(twist))
    airfoil = getXsecAirfoil(wing_id, xsec, increment)
    return WingXSec(xyz_le, chord, twist, airfoil=airfoil)

# determine the airfoil type
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

# x rotation of a point
def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

# y rotation of a point
def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

# z rotation of a point
def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

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
