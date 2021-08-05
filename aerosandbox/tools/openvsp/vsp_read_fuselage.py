import aerosandbox
from aerosandbox.geometry import Fuselage
from aerosandbox.geometry import FuselageXSec
import openvsp as vsp
import aerosandbox.numpy as np
from ctypes import *

# ----------------------------------------------------------------------
#  vsp read fuselage
# ----------------------------------------------------------------------

def vsp_read_fuselage(fuselage_id):
    """This reads an OpenVSP fuselage geometry and writes it to a aerosandbox fuselage format.

    Assumptions:
    1. OpenVSP fuselage is "conventionally shaped" (generally narrow at nose and tail, wider in center). 
    2. Fuselage is designed in VSP as it appears in real life. That is, the VSP model does not rely on
       superficial elements such as canopies, stacks, or additional fuselages to cover up internal lofting oddities.
    3. This program will NOT account for multiple geometries comprising the fuselage. For example: a wingbox mounted beneath
       is a separate geometry and will NOT be processed.
    4. Fuselage origin is located at nose. VSP file origin can be located anywhere, preferably at the forward tip
       of the vehicle or in front (to make all X-coordinates of vehicle positive).
    5. Written for OpenVSP 3.21.1
    
    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. VSP 10-digit geom ID for fuselage.
    
    Outputs:
    aerosandbox fuselage   
        """      
    
    print("Converting fuselage: " + fuselage_id)
    # get total length of fuselage
    total_length = vsp.GetParmVal(fuselage_id, 'Length', 'Design')

    # read the xsec data
    xsec_root_id = vsp.GetXSecSurf(fuselage_id, 0)
    xsec_num = vsp.GetNumXSec(xsec_root_id)
    xsecs = []
    for increment in range(0, xsec_num):
        xsecs.append(getVspXSec(xsec_root_id, xsec_num, total_length, increment))
    # get the name
    if vsp.GetGeomName(fuselage_id):
        tag = vsp.GetGeomName(fuselage_id)
    else:
        tag = 'FuselageGeom'
    # get leading edge
    xyz_le = np.array([0, 0, 0])
    xyz_le[0] = vsp.GetParmVal(fuselage_id, 'X_Location', 'XForm')
    xyz_le[1] = vsp.GetParmVal(fuselage_id, 'Y_Location', 'XForm')
    xyz_le[2] = vsp.GetParmVal(fuselage_id, 'Z_Location', 'XForm')
    # create the fuselage
    return Fuselage(tag, xyz_le, xsecs)    
    
# Get Fuselage segments
def getVspXSec(xsec_root_id, xsec_num, total_length, increment):
    print("   Processing xsec: " + str(increment))
    # Create the segment
    #
    # this is a point on the fuse and not the center point so entire geometry is shifted for now
    point = vsp.ComputeXSecPnt(xsec, 0.0)    # get xsec point at leading edge of tip chord
    p = (c_double * 3).from_address(int(np.ctypeslib.as_array(point.data())))
    xyz_c =  np.array(list(p))
    print("   fuse test: " + str(xyz_c))

    xsec   = vsp.GetXSec(xsec_root_id, increment) # VSP XSec ID.
    X_Loc_P = vsp.GetXSecParm(xsec, 'XLocPercent')
    Y_Loc_P = vsp.GetXSecParm(xsec, 'YLocPercent')
    Z_Loc_P = vsp.GetXSecParm(xsec, 'ZLocPercent')
    
    height             = vsp.GetXSecHeight(xsec)
    width              = vsp.GetXSecWidth(xsec)
    percent_x_location = vsp.GetParmVal(X_Loc_P) # Along fuselage length.
    #xyz_c[0] = percent_x_location*total_length
    percent_y_location = vsp.GetParmVal(Y_Loc_P)
    percent_z_location = vsp.GetParmVal(Z_Loc_P ) # Vertical deviation of fuselage center.
    effective_diameter = (height+width)/2. 
    radius = effective_diameter/2.

    #xsec shape not supported for aerosandbox FuselageXSec
    #shape       = vsp.GetXSecShape(segment.vsp_data.xsec_id)
    #shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
    #vsp_data.shape = shape_dict[shape]    
    return FuselageXSec(xyz_c, radius)
    

def get_fuselage_height(fuselage, location):
    """This linearly estimates fuselage height at any percentage point (0,100) along fuselage length.
    
    Assumptions:
    Written for OpenVSP 3.16.1
    
    Source:
    N/A

    Inputs:
    0. Pre-loaded VSP vehicle in memory, via vsp_read.
    1. Suave fuselage [object], containing fuselage.vsp_data.xsec_num in its data structure.
    2. Fuselage percentage point [float].
    
    Outputs:
    height [m]
    """
    for jj in range(1, fuselage.vsp_data.xsec_num):        # Begin at second section, working toward tail.
        if fuselage.Segments[jj].percent_x_location>=location and fuselage.Segments[jj-1].percent_x_location<location:  
            # Find two sections on either side (or including) the desired fuselage length percentage.
            a        = fuselage.Segments[jj].percent_x_location                            
            b        = fuselage.Segments[jj-1].percent_x_location
            a_height = fuselage.Segments[jj].height        # Linear approximation.
            b_height = fuselage.Segments[jj-1].height
            slope    = (a_height - b_height)/(a-b)
            height   = ((location-b)*(slope)) + (b_height)    
            break
    return height
