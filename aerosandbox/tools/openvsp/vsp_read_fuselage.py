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

def vsp_read_fuselage(fuselage_id, units_type='SI', fineness=True):
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
	2. Units_type set to 'SI' (default) or 'Imperial'.
	3. Boolean for whether or not to compute fuselage finenesses (default = True).
	
	Outputs:
	Writes aerosandbox fuselage, with these geometries:           (all defaults are SI, but user may specify Imperial)

		Fuselages.Fuselage.			
			origin                                  [m] in all three dimensions
			width                                   [m]
			lengths.
			  total                                 [m]
			  nose                                  [m]
			  tail                                  [m]
			heights.
			  maximum                               [m]
			  at_quarter_length                     [m]
			  at_three_quarters_length              [m]
			effective_diameter                      [m]
			fineness.nose                           [-] ratio of nose section length to fuselage effective diameter
			fineness.tail                           [-] ratio of tail section length to fuselage effective diameter
			areas.wetted                            [m^2]
			tag                                     <string>
			segment[].   (segments are in ordered container and callable by number)
			  vsp.shape                               [point,circle,round_rect,general_fuse,fuse_file]
			  vsp.xsec_id                             <10 digit string>
			  percent_x_location
			  percent_z_location
			  height
			  width
			  length
			  effective_diameter
			  tag
			vsp.xsec_num                              <integer of fuselage segment quantity>
			vsp.xsec_surf_id                          <10 digit string>

	Properties Used:
	N/A
	"""  	
	fuselage = aerosandbox.geometry.Fuselage()	
	
	if units_type == 'SI':
		units_factor = Units.meter * 1.
	elif units_type == 'imperial':
		units_factor = Units.foot * 1.
	elif units_type == 'inches':
		units_factor = Units.inch * 1.	
		
	if vsp.GetGeomName(fuselage_id):
		fuselage.tag = vsp.GetGeomName(fuselage_id)
	else: 
		fuselage.tag = 'FuselageGeom'

	fuselage.origin[0][0] = vsp.GetParmVal(fuselage_id, 'X_Location', 'XForm') * units_factor
	fuselage.origin[0][1] = vsp.GetParmVal(fuselage_id, 'Y_Location', 'XForm') * units_factor
	fuselage.origin[0][2] = vsp.GetParmVal(fuselage_id, 'Z_Location', 'XForm') * units_factor

	#fuselage.lengths.total         = vsp.GetParmVal(fuselage_id, 'Length', 'Design') * units_factor
	#fuselage.vsp_data.xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0) 			# There is only one XSecSurf in geom.
	fuselage.vsp_data.xsec_num     = vsp.GetNumXSec(fuselage.vsp_data.xsec_surf_id) 		# Number of xsecs in fuselage.	
	
	x_locs    = []
	heights   = []
	widths    = []
	eff_diams = []
	lengths   = []
	
	# -----------------
	# Fuselage segments
	# -----------------
	
	for ii in range(0, fuselage.vsp_data.xsec_num):
		
		# Create the segment
		x_sec                     = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii) # VSP XSec ID.
		segment                   = aerosandbox.geometry.FuselageXSec()
		segment.vsp_data.xsec_id  = x_sec 
		segment.tag               = 'segment_' + str(ii)
		
		# Pull out Parms that will be needed
		X_Loc_P = vsp.GetXSecParm(x_sec, 'XLocPercent')
		Z_Loc_P = vsp.GetXSecParm(x_sec, 'ZLocPercent')
		
		segment.percent_x_location = vsp.GetParmVal(X_Loc_P) # Along fuselage length.
		segment.percent_z_location = vsp.GetParmVal(Z_Loc_P ) # Vertical deviation of fuselage center.
		segment.height             = vsp.GetXSecHeight(segment.vsp_data.xsec_id) * units_factor
		segment.width              = vsp.GetXSecWidth(segment.vsp_data.xsec_id) * units_factor
		segment.effective_diameter = (segment.height+segment.width)/2. 
		
		x_locs.append(segment.percent_x_location)	 # Save into arrays for later computation.
		heights.append(segment.height)
		widths.append(segment.width)
		eff_diams.append(segment.effective_diameter)
		
		if ii != (fuselage.vsp_data.xsec_num-1): # Segment length: stored as length since previous segment. (last segment will have length 0.0.)
			next_xsec = vsp.GetXSec(fuselage.vsp_data.xsec_surf_id, ii+1)
			X_Loc_P_p = vsp.GetXSecParm(next_xsec, 'XLocPercent')
			percent_x_loc_p1 = vsp.GetParmVal(X_Loc_P_p) 
			segment.length = fuselage.lengths.total*(percent_x_loc_p1 - segment.percent_x_location) * units_factor
		else:
			segment.length = 0.0
		lengths.append(segment.length)
		
		shape	   = vsp.GetXSecShape(segment.vsp_data.xsec_id)
		shape_dict = {0:'point',1:'circle',2:'ellipse',3:'super ellipse',4:'rounded rectangle',5:'general fuse',6:'fuse file'}
		segment.vsp_data.shape = shape_dict[shape]	
	
		fuselage.Segments.append(segment)

	fuselage.heights.at_quarter_length          = get_fuselage_height(fuselage, .25)  # Calls get_fuselage_height function (below).
	fuselage.heights.at_three_quarters_length   = get_fuselage_height(fuselage, .75) 
	fuselage.heights.at_wing_root_quarter_chord = get_fuselage_height(fuselage, .4) 

	fuselage.heights.maximum    = max(heights) 		# Max segment height.	
	fuselage.width		    = max(widths) 		# Max segment width.
	fuselage.effective_diameter = max(eff_diams)		# Max segment effective diam.
	
	fuselage.areas.front_projected  = np.pi*((fuselage.effective_diameter)/2)**2

	eff_diam_gradients_fwd = np.array(eff_diams[1:]) - np.array(eff_diams[:-1])		# Compute gradients of segment effective diameters.
	eff_diam_gradients_fwd = np.multiply(eff_diam_gradients_fwd, lengths[:-1])
		

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
