# Created:  Jun 2018, T. St Francis
# Modified: Aug 2021  Michael Shamberger
#           Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald
#           Jul 2020, E. Botero
# Original code taken from Suave project and modified for AeroSandbox

import aerosandbox
from aerosandbox.geometry import Airplane
from aerosandbox.tools.openvsp.vsp_read_fuselage import vsp_read_fuselage
from aerosandbox.tools.openvsp.vsp_read_wing import vsp_read_wing
import aerosandbox.numpy as np

import openvsp as vsp


# ----------------------------------------------------------------------
#  vsp read
# ----------------------------------------------------------------------
def vsp_read(tag):     
    """This reads an OpenVSP vehicle geometry and writes it into a Aerosandbox vehicle format.
    Includes wings, fuselages, and propellers.

    Assumptions:
    1. OpenVSP vehicle is composed of conventionally shaped fuselages, wings, and propellers. 
    1a. OpenVSP fuselage: generally narrow at nose and tail, wider in center). 
    1b. Fuselage is designed in VSP as it appears in real life. That is, the VSP model does not rely on
       superficial elements such as canopies, stacks, or additional fuselages to cover up internal lofting oddities.
    1c. This program will NOT account for multiple geometries comprising the fuselage. For example: a wingbox mounted beneath
       is a separate geometry and will NOT be processed.
    2. Fuselage origin is located at nose. VSP file origin can be located anywhere, preferably at the forward tip
       of the vehicle or in front (to make all X-coordinates of vehicle positive).
    3. Written for OpenVSP 3.24
    
    Source:
    N/A

    Inputs:
    1. A tag for an XML file in format .vsp3.

    Outputs:
    Writes Aerosandbox vehicle

    Properties Used:
    N/A
    """      
    
    vsp.ClearVSPModel() 
    vsp.ReadVSPFile(tag)    
    
    vsp_fuselages = []
    vsp_wings     = []    
    vsp_props     = []
    vsp_geoms     = vsp.FindGeoms()
    geom_names    = []

    # The two for-loops below are in anticipation of an OpenVSP API update with a call for GETGEOMTYPE.
    # This print function allows user to enter VSP GeomID manually as first argument in vsp_read functions.
    print("VSP geometry IDs: ")    
    
    # Label each geom type by storing its VSP geom ID. 
    for geom in vsp_geoms: 
        geom_name = vsp.GetGeomName(geom)
        geom_names.append(geom_name)
        print(str(geom_name) + ': ' + geom)
    
    # --------------------------------
    # AUTOMATIC VSP ENTRY & PROCESSING
    # --------------------------------        
    for geom in vsp_geoms:
        geom_name = vsp.GetGeomTypeName(str(geom))
        
        if geom_name == 'Fuselage':
            vsp_fuselages.append(geom)
        if geom_name == 'Wing':
            vsp_wings.append(geom)
        # No aerosandbox propeller geometry class available
        #if geom_name == 'Propeller':
        #    vsp_props.append(geom)
    
    #Read VSP geoms and store in Aerosandbox components
    xyz_ref = np.array([0, 0, 0])
    fuselages = []
    for fuselage_id in vsp_fuselages:
        fuselages.append(vsp_read_fuselage(fuselage_id))
        
    wings = []
    for wing_id in vsp_wings:
        wings.append(vsp_read_wing(wing_id))
    
    return Airplane(tag, xyz_ref, wings, fuselages)
