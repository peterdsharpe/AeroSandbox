from aerosandbox_legacy_v0 import *



nlf414f_il = Airfoil(name = "nlf414f-il", coordinates = np.array([
[1.0, 0.0001959], 
[0.9979467, 0.0003898], 
[0.9919964, 0.0010932],
[0.9823112, 0.0023427],
[0.9689228, 0.0042629],
[0.9520135, 0.007013],
[0.9316924, 0.0107129],
[0.9082243, 0.0155438],
[0.882182, 0.0216244],
[0.8539022, 0.0291099],
[0.8241454, 0.0379971],
[0.7937308, 0.0478613],
[0.763343, 0.0580612],
[0.7335511, 0.0675314],
[0.7038118, 0.075116],
[0.6727199, 0.0808877],
[0.6402998, 0.0853921],
[0.6067503, 0.0889526],
[0.5722421, 0.0916209],
[0.537091, 0.0934517],
[0.5015802, 0.0944966],
[0.4658635, 0.0947739],
[0.430091, 0.0943827],
[0.3945049, 0.0933135],
[0.3594246, 0.0916002],
[0.3249547, 0.0892465],
[0.291342, 0.0863393],
[0.2587907, 0.0829001],
[0.2274915, 0.07895],
[0.197592, 0.0745702],
[0.1692863, 0.0698309],
[0.1426732, 0.0648149],
[0.1179169, 0.0596196],
[0.0951302, 0.054187],
[0.0744421, 0.0485159],
[0.0560369, 0.042623],
[0.0400431, 0.0365741],
[0.0265241, 0.0304597],
[0.0156058, 0.0242425],
[0.0075199, 0.0177575],
[0.0024032, 0.0106837],
[0.0002094, 0.0032797],
[0.0, 0.0],
 [0.0002217, -0.002114],
 [0.0022841, -0.0069216],
 [0.0074238, -0.0107903],
 [0.0159347, -0.0144468],
 [0.0278726, -0.0180257],
 [0.04298, -0.02148],
 [0.0609524, -0.024822],
 [0.0816471, -0.0280287],
 [0.1049076, -0.0310793],
 [0.1305626, -0.0339305],
 [0.1584566, -0.0365595],
 [0.1883933, -0.0389607],
 [0.2201672, -0.0411107],
 [0.2535623, -0.0429745],
 [0.2883051, -0.0445354],
 [0.3241533, -0.0457801],
 [0.3608882, -0.0466806],
 [0.3983647, -0.0471698],
 [0.4363414, -0.0472223],
 [0.4745037, -0.0468228],
 [0.5126616, -0.0460074],
 [0.5505865, -0.0447514],
 [0.5879576, -0.0429205],
 [0.624506, -0.0404731],
 [0.6599402, -0.0372181],
 [0.694131, -0.032883],
 [0.7271526, -0.0262009],
 [0.7621386, -0.0180301],
 [0.797513, -0.0117427],
 [0.8315642, -0.0070185],
 [0.863597, -0.0036384],
 [0.8928488, -0.0015343],
 [0.9190598, -0.0004323],
 [0.941719, 3.67e-05],
 [0.9606961, 0.0001832],
 [0.9759963, 0.0001791],
 [0.9876149, 0.0001215],
 [0.9955088, -1.58e-05],
 [1.000011, -0.0001959]
 ] ))




wildTurkey = Airplane(
    name="John Parker 1977 Wild Turkey",
    xyz_ref=[0, 0, 0], # CG location
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0], # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=[ # The wing's cross ("X") sections
                WingXSec(  # Buried Wing
                    xyz_le=[0, 0, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=1.2,
                    twist=0, # degrees
                    airfoil=nlf414f_il,
                    # control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                    # control_surface_deflection=0, # degrees
                    # control_surface_hinge_point=0.75 # as chord fraction
                ),
                WingXSec(  # Fillet Root
                    xyz_le=[0, 0.1, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=1.2,
                    twist=0, # degrees
                    airfoil=nlf414f_il,
                    # control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                    # control_surface_deflection=0, # degrees
                    # control_surface_hinge_point=0.75 # as chord fraction
                ),
                WingXSec(  # Fillet Tip
                    xyz_le=[0.4, 0.3, 0], # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.2,
                    twist=0, # degrees
                    airfoil=nlf414f_il,
                    # control_surface_type='symmetric',  # Flap # Control surfaces are applied between a given XSec and the next one.
                    # control_surface_deflection=0, # degrees
                    # control_surface_hinge_point=0.75 # as chord fraction
                ),
                WingXSec(  # Mid
                    xyz_le=[0.4, 0.9, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=nlf414f_il,
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # Tip
                    xyz_le=[0.4, 1.3, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=nlf414f_il,
                )
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            xyz_le=[1.5, 0, 0.1],
            symmetric=True,
            xsecs=[
                WingXSec(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=-10,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Elevator
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # tip
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist=-10,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        ),
        Wing(
            name="Vertical Stabilizer",
            xyz_le=[1.5, 0, 0.1],
            symmetric=False,
            xsecs=[
                WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Rudder
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)

aero_problem = vlm3( # Analysis type: Vortex Lattice Method, version 3
    airplane=wildTurkey,
    op_point=OperatingPoint(
        velocity=10,
        alpha=5,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
)

aero_problem.run() # Runs and prints results to console
aero_problem.draw() # Creates an interactive display of the surface pressures and streamlines

for wing in wildTurkey.wings:
    print(wing.name)
    print("\tSpan {:4}m & Aspect Ratio {:4}".format(wing.span(),wing.aspect_ratio()))
    print("\tWetted area {}m^2".format(wing.area_wetted()))
    print("\tFrontal area {}m^2".format(wing.area_projected()))
