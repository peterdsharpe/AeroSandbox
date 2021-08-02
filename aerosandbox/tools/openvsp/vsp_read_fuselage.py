## @ingroup Input_Output-OpenVSP
# vsp_read_fuselage.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import aerosandbox
from aerosandbox.geometry import Fuselage
import openvsp as vsp
import aerosandbox.numpy as np

# ----------------------------------------------------------------------
#  vsp read fuselage
# ----------------------------------------------------------------------

def vsp_read_fuselage(fuselage_id, units_type='SI'):
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

        # read the xsec data
        fuselage.vsp_data.xsec_num     = vsp.GetNumXSec(fuselage.vsp_data.xsec_surf_id)                 # Number of xsecs in fuselage.
        xsec = []
        for i in range(0, fuselage.vsp_data.xsec_num):
            xsec.append(getXsec(i))
        # get the name
        if vsp.GetGeomName(fuselage_id):
            fuselage.tag = vsp.GetGeomName(fuselage_id)
        else:
            fuselage.tag = 'FuselageGeom'
        # get leading edge
        xyz_le = np.array([0, 0, 0])
        xyz_le[0] = vsp.GetParmVal(fuselage_id, 'X_Location', 'XForm')
        xyz_le[1] = vsp.GetParmVal(fuselage_id, 'Y_Location', 'XForm')
        xyz_le[2] = vsp.GetParmVal(fuselage_id, 'Z_Location', 'XForm')
        # create the fuselage
	fuselage = aerosandbox.geometry.Fuselage(fuselage.tag, xyz_le, xsec)	
	
# Get Fuselage segments
def getVspXSec(i):
	# Create the segment
        xyz_c = np.array([0, 0, 0])
	x_sec   = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii) # VSP XSec ID.
	X_Loc_P = vsp.GetXSecParm(x_sec, 'XLocPercent')
	Z_Loc_P = vsp.GetXSecParm(x_sec, 'ZLocPercent')
        radius = vsp.GetXSecParm(x_sec, 'Circle_Diameter')/2
	
	segment.percent_x_location = vsp.GetParmVal(X_Loc_P) # Along fuselage length.
	segment.percent_z_location = vsp.GetParmVal(Z_Loc_P ) # Vertical deviation of fuselage center.
	segment.height             = vsp.GetXSecHeight(segment.vsp_data.xsec_id)
	segment.width              = vsp.GetXSecWidth(segment.vsp_data.xsec_id)
	segment.effective_diameter = (segment.height+segment.width)/2. 
		
	if ii != (fuselage.vsp_data.xsec_num-1): # Segment length: stored as length since previous segment. (last segment will have length 0.0.)
		next_xsec = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii+1)
		X_Loc_P_p = vsp.GetXSecParm(next_xsec, 'XLocPercent')
		percent_x_loc_p1 = vsp.GetParmVal(X_Loc_P_p) 
		segment.length = fuselage.lengths.total*(percent_x_loc_p1 - segment.percent_x_location)
	else:
		segment.length = 0.0
	lengths.append(segment.length)
	
        #xsec shape not supported for aerosandbox FuselageXSec
	#shape	   = vsp.GetXSecShape(segment.vsp_data.xsec_id)
	#shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
	#vsp_data.shape = shape_dict[shape]	
	
        return FuselageXSec(xyz_c, radius)

return fuselage
	

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
	
	Properties Used:
	N/A
	"""
	for jj in range(1, fuselage.vsp_data.xsec_num):		# Begin at second section, working toward tail.
		if fuselage.Segments[jj].percent_x_location>=location and fuselage.Segments[jj-1].percent_x_location<location:  
			# Find two sections on either side (or including) the desired fuselage length percentage.
			a        = fuselage.Segments[jj].percent_x_location							
			b        = fuselage.Segments[jj-1].percent_x_location
			a_height = fuselage.Segments[jj].height		# Linear approximation.
			b_height = fuselage.Segments[jj-1].height
			slope    = (a_height - b_height)/(a-b)
			height   = ((location-b)*(slope)) + (b_height)	
			break
	return height
