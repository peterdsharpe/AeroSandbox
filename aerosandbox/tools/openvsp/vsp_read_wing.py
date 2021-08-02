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
def vsp_read_wing(wing_id, units_type='SI',write_airfoil_file=True): 	
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
	Writes aerosandbox wing object, with these geometries, from VSP:
		Wings.Wing.    (* is all keys)
			origin                                  [m] in all three dimensions
			spans.projected                         [m]
			chords.root                             [m]
			chords.tip                              [m]
			aspect_ratio                            [-]
			sweeps.quarter_chord                    [radians]
			twists.root                             [radians]
			twists.tip                              [radians]
			thickness_to_chord                      [-]
			dihedral                                [radians]
			symmetric                               <boolean>
			tag                                     <string>
			areas.exposed                           [m^2]
			areas.reference                         [m^2]
			areas.wetted                            [m^2]
			Segments.
			  tag                                   <string>
			  twist                                 [radians]
			  percent_span_location                 [-]  .1 is 10%
			  root_chord_percent                    [-]  .1 is 10%
			  dihedral_outboard                     [radians]
			  sweeps.quarter_chord                  [radians]
			  thickness_to_chord                    [-]
			  airfoil                               <NACA 4-series, 6 series, or airfoil file>

	Properties Used:
	N/A
	"""  
	
	# Check if this is vertical tail, this seems like a weird first step but it's necessary
	# Get the initial rotation to get the dihedral angles
	#x_rot = vsp.GetParmVal( wing_id,'X_Rotation','XForm')		
	#if  x_rot >=70:
	#	wing = aerosandbox.Components.Wings.Vertical_Tail()
	#	wing.vertical = True
	#	x_rot = (90-x_rot) * Units.deg
	#else:
		# Instantiate a wing
	wing = aerosandbox.geometry.Wing()
	
	# Set the units
	if units_type == 'SI':
		units_factor = Units.meter * 1.
	elif units_type == 'imperial':
		units_factor = Units.foot * 1.
	elif units_type == 'inches':
		units_factor = Units.inch * 1.		

	# Apply a tag to the wing
	if vsp.GetGeomName(wing_id):
		tag = vsp.GetGeomName(wing_id)
		tag = tag.translate(t_table)
		wing.tag = tag
	else: 
		wing.tag = 'winggeom'
	
	# Top level wing parameters
	# Wing origin
	wing.origin[0][0] = vsp.GetParmVal(wing_id, 'X_Location', 'XForm') * units_factor 
	wing.origin[0][1] = vsp.GetParmVal(wing_id, 'Y_Location', 'XForm') * units_factor 
	wing.origin[0][2] = vsp.GetParmVal(wing_id, 'Z_Location', 'XForm') * units_factor 
	
	# Wing Symmetry
	sym_planar = vsp.GetParmVal(wing_id, 'Sym_Planar_Flag', 'Sym')
	sym_origin = vsp.GetParmVal(wing_id, 'Sym_Ancestor', 'Sym')

	# Check for symmetry
	if sym_planar == 2. and sym_origin == 1.: #origin at wing, not vehicle
		wing.symmetric = True	
	else:
		wing.symmetric = False
		

	
	#More top level parameters
	#total_proj_span      = vsp.GetParmVal(wing_id, 'TotalProjectedSpan', 'WingGeom') * units_factor
	#wing.aspect_ratio    = vsp.GetParmVal(wing_id, 'TotalAR', 'WingGeom')
	#wing.areas.reference = vsp.GetParmVal(wing_id, 'TotalArea', 'WingGeom') * units_factor**2 
	#wing.spans.projected = total_proj_span 
 
	# Check if this is a single segment wing
	xsec_surf_id      = vsp.GetXSecSurf(wing_id, 0)   # This is how VSP stores surfaces.
	x_sec_1           = vsp.GetXSec(xsec_surf_id, 1)
	x_sec_1_span_parm = vsp.GetXSecParm(x_sec_1,'Span')
	x_sec_1_span      = vsp.GetParmVal(x_sec_1_span_parm)*(1+wing.symmetric) * units_factor
	
	if x_sec_1_span == wing.spans.projected:
		single_seg = True
	else:
		single_seg = False

	segment_num       = vsp.GetNumXSec(xsec_surf_id)	# Get number of wing segments (is one more than the VSP GUI shows).
	x_sec             = vsp.GetXSec(xsec_surf_id, 0)
	chord_parm        = vsp.GetXSecParm(x_sec,'Root_Chord')
	
	total_chord      = vsp.GetParmVal(chord_parm) 

	span_sum         = 0.				# Non-projected.
	proj_span_sum    = 0.				# Projected.
	segment_spans    = [None] * (segment_num) 	# Non-projected.
	segment_dihedral = [None] * (segment_num)
	segment_sweeps_quarter_chord = [None] * (segment_num)
	
	# Check for wing segment *inside* fuselage, then skip XSec_0 to start at first exposed segment.
	if np.isclose(total_chord,1):
		start = 1
		xsec_surf_id = vsp.GetXSecSurf(wing_id, 1)	
		x_sec        = vsp.GetXSec(xsec_surf_id, 0)
		chord_parm   = vsp.GetXSecParm(x_sec,'Tip_Chord')
		root_chord   = vsp.GetParmVal(chord_parm)* units_factor
	else:
		start = 0
		root_chord = total_chord * units_factor
	
	# -------------
	# Wing segments
	# -------------
	
	if single_seg == False:
		
		# Convert VSP XSecs to aerosandbox segments. (Wing segments are defined by outboard sections in VSP, but inboard sections in aerosandbox.) 
		for i in range(start, segment_num+1):	
			# XSec airfoil
			if start!=0:
				jj = i-1  # Airfoil index i-1 because VSP airfoils and sections are one index off relative to aerosandbox.
			else:
				jj= i*1			
			segment = aerosandbox.Components.Wings.Segment()
			segment.tag                   = 'Section_' + str(i)
			thick_cord                    = vsp.GetParmVal(wing_id, 'ThickChord', 'XSecCurve_' + str(jj))
			segment.thickness_to_chord    = thick_cord	# Thick_cord stored for use in airfoil, below.		
			if i!=segment_num:
				segment_root_chord    = vsp.GetParmVal(wing_id, 'Root_Chord', 'XSec_' + str(i)) * units_factor
			else:
				segment_root_chord    = 0.0
			segment.root_chord_percent    = segment_root_chord / root_chord		
			segment.percent_span_location = proj_span_sum / (total_proj_span/(1+wing.symmetric))
			segment.twist                 = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(jj)) * Units.deg
			
			if i==start:
				wing.thickness_to_chord = thick_cord
		
			if i < segment_num:      # This excludes the tip xsec, but we need a segment in aerosandbox to store airfoil.
				sweep     = vsp.GetParmVal(wing_id, 'Sweep', 'XSec_' + str(i)) * Units.deg
				sweep_loc = vsp.GetParmVal(wing_id, 'Sweep_Location', 'XSec_' + str(i))
				AR        = 2*vsp.GetParmVal(wing_id, 'Aspect', 'XSec_' + str(i))
				taper     = vsp.GetParmVal(wing_id, 'Taper', 'XSec_' + str(i))
				   
				segment_sweeps_quarter_chord[i] = convert_sweep(sweep,sweep_loc,0.25,AR,taper)
				segment.sweeps.quarter_chord    = segment_sweeps_quarter_chord[i]  # Used again, below
				
				# Used for dihedral computation, below.
				segment_dihedral[i]	      = vsp.GetParmVal(wing_id, 'Dihedral', 'XSec_' + str(i)) * Units.deg  + x_rot
				segment.dihedral_outboard     = segment_dihedral[i]
		
				segment_spans[i] 	      = vsp.GetParmVal(wing_id, 'Span', 'XSec_' + str(i)) * units_factor
				proj_span_sum += segment_spans[i] * np.cos(segment_dihedral[i])	
				span_sum      += segment_spans[i]
			else:
				segment.root_chord_percent    = (vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(i-1))) * units_factor /root_chord
		

			xsec_id = str(vsp.GetXSec(xsec_surf_id, jj))
			airfoil = Airfoil()
			if vsp.GetXSecShape(xsec_id) == vsp.XS_FOUR_SERIES: 	# XSec shape: NACA 4-series
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
		
			elif vsp.GetXSecShape(xsec_id) == vsp.XS_SIX_SERIES: 	# XSec shape: NACA 6-series
				thick_cord_round = int(np.around(thick_cord*100))
				a_value          = vsp.GetParmVal(wing_id, 'A', 'XSecCurve_' + str(jj))
				ideal_CL         = int(np.around(vsp.GetParmVal(wing_id, 'IdealCl', 'XSecCurve_' + str(jj))*10))
				series_vsp       = int(vsp.GetParmVal(wing_id, 'Series', 'XSecCurve_' + str(jj)))
				series_dict      = {0:'63',1:'64',2:'65',3:'66',4:'67',5:'63A',6:'64A',7:'65A'} # VSP series values.
				series           = series_dict[series_vsp]
				airfoil.tag      = 'NACA ' + series + str(ideal_CL) + str(thick_cord_round) + ' a=' + str(np.around(a_value,1))			
					
		
			elif vsp.GetXSecShape(xsec_id) == vsp.XS_FILE_AIRFOIL:	# XSec shape: 12 is type AF_FILE
				airfoil.thickness_to_chord = thick_cord
				# VSP airfoil API calls get coordinates and write files with the final argument being the fraction of segment position, regardless of relative spans. 
				# (Write the root airfoil with final arg = 0. Write 4th airfoil of 5 segments with final arg = .8)
				
			if write_airfoil_file==True:
				vsp.WriteSeligAirfoil(str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat', wing_id, float(jj/segment_num))
				airfoil.coordinate_file    = str(wing.tag) + '_airfoil_XSec_' + str(jj) +'.dat'
				airfoil.tag                = 'airfoil'	
		
				segment.append_airfoil(airfoil)
		
			wing.Segments.append(segment)
		
		# Wing dihedral 
		proj_span_sum_alt = 0.
		span_sum_alt      = 0.
		sweeps_sum        = 0.			
		
		for ii in range(start, segment_num):
			span_sum_alt += segment_spans[ii]
			proj_span_sum_alt += segment_spans[ii] * np.cos(segment_dihedral[ii])  # Use projected span to find total wing dihedral.
			sweeps_sum += segment_spans[ii] * np.tan(segment_sweeps_quarter_chord[ii])	
		
		wing.dihedral              = np.arccos(proj_span_sum_alt / span_sum_alt) 
		wing.sweeps.quarter_chord  = -np.arctan(sweeps_sum / span_sum_alt)  # Minus sign makes it positive sweep.
		
		# Add a tip segment, all values are zero except the tip chord
		tc = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_' + str(segment_num-1)) * units_factor
		
		# Chords
		wing.chords.root              = vsp.GetParmVal(wing_id, 'Tip_Chord', 'XSec_0') * units_factor
		wing.chords.tip               = tc
		wing.chords.mean_geometric    = wing.areas.reference / wing.spans.projected
		
		# Just double calculate and fix things:
		wing = wing_segmented_planform(wing)
				
			
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
		sweep     = vsp.GetParmVal(x_sec_1_sweep_parm) * Units.deg
		sweep_loc = vsp.GetParmVal(x_sec_1_sweep_loc_parm)
		taper     = vsp.GetParmVal(x_sec_1_taper_parm)
		c_4_sweep = convert_sweep(sweep,sweep_loc,0.25,wing.aspect_ratio,taper)		
		
		# Pull and pack
		wing.sweeps.quarter_chord  = c_4_sweep
		wing.taper                 = taper
		wing.dihedral              = vsp.GetParmVal(x_sec_1_dih_parm) * Units.deg + x_rot
		wing.chords.root           = vsp.GetParmVal(x_sec_1_rc_parm)* units_factor
		wing.chords.tip            = vsp.GetParmVal(x_sec_1_tc_parm) * units_factor	
		wing.chords.mean_geometric = wing.areas.reference / wing.spans.projected
		
		# Just double calculate and fix things:
		wing = wing_planform(wing)		


	# Twists
	wing.twists.root      = vsp.GetParmVal(wing_id, 'Twist', 'XSec_0') * Units.deg
	wing.twists.tip       = vsp.GetParmVal(wing_id, 'Twist', 'XSec_' + str(segment_num-1)) * Units.deg
	
	
	return wing


def convert_sweep(sweep,sweep_loc,new_sweep_loc,AR,taper):
	
	sweep_LE = np.arctan(np.tan(sweep)+4*sweep_loc*
                              (1-taper)/(AR*(1+taper))) 
	
	new_sweep = np.arctan(np.tan(sweep_LE)-4*new_sweep_loc*
                          (1-taper)/(AR*(1+taper))) 
	
	return new_sweep
