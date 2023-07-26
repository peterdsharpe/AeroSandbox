import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from typing import Union
import pytest


def a() -> asb.Airplane:

    fuselage_cabin_diameter = 20.4 * u.foot
    fuselage_cabin_radius = fuselage_cabin_diameter / 2
    fuselage_cabin_xsec_area = np.pi * fuselage_cabin_radius ** 2

    fuselage_cabin_length = 123.2 * u.foot
    fwd_fuel_tank_length = 6
    aft_fuel_tank_length = fwd_fuel_tank_length

    # Compute x-locations of various fuselage stations
    nose_fineness_ratio = 1.67
    tail_fineness_ratio = 2.62

    x_nose = 0
    x_nose_to_fwd_tank = x_nose + nose_fineness_ratio * fuselage_cabin_diameter
    x_fwd_tank_to_cabin = x_nose_to_fwd_tank + fwd_fuel_tank_length
    x_cabin_to_aft_tank = x_fwd_tank_to_cabin + fuselage_cabin_length
    x_aft_tank_to_tail = x_cabin_to_aft_tank + aft_fuel_tank_length
    x_tail = x_aft_tank_to_tail + tail_fineness_ratio * fuselage_cabin_diameter

    # Build up the actual fuselage nodes
    x_fuse_sections = []
    z_fuse_sections = []
    r_fuse_sections = []

    def linear_map(
            f_in: Union[float, np.ndarray],
            min_in: Union[float, np.ndarray],
            max_in: Union[float, np.ndarray],
            min_out: Union[float, np.ndarray],
            max_out: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Linearly maps an input `f_in` from range (`min_in`, `max_in`) to (`min_out`, `max_out`).

        Args:
            f_in: Input value
            min_in:
            max_in:
            min_out:
            max_out:

        Returns:
            f_out: Output value

        """
        # if min_in == 0 and max_in == 1:
        #     f_nondim = f_in
        # else:
        f_nondim = (f_in - min_in) / (max_in - min_in)

        # if max_out == 0 and min_out == 1:
        #     f_out = f_nondim
        # else:
        f_out = f_nondim * (max_out - min_out) + min_out

        return f_out

    # Nose
    x_sect_nondim = np.sinspace(0, 1, 10)
    z_sect_nondim = -0.3 * (1 - x_sect_nondim) ** 2
    r_sect_nondim = (1 - (1 - x_sect_nondim) ** 2) ** 0.5

    x_fuse_sections.append(
        linear_map(
            f_in=x_sect_nondim,
            min_in=0, max_in=1,
            min_out=x_nose, max_out=x_nose_to_fwd_tank
        )
    )
    z_fuse_sections.append(
        linear_map(
            f_in=z_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )
    r_fuse_sections.append(
        linear_map(
            f_in=r_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )

    # Fwd tank
    x_sect_nondim = np.linspace(0, 1, 2)
    z_sect_nondim = np.zeros_like(x_sect_nondim)
    r_sect_nondim = np.ones_like(x_sect_nondim)

    x_fuse_sections.append(
        linear_map(
            f_in=x_sect_nondim,
            min_in=0, max_in=1,
            min_out=x_nose_to_fwd_tank, max_out=x_fwd_tank_to_cabin
        )
    )
    z_fuse_sections.append(
        linear_map(
            f_in=z_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )
    r_fuse_sections.append(
        linear_map(
            f_in=r_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )

    # Cabin
    x_sect_nondim = np.linspace(0, 1, 2)
    z_sect_nondim = np.zeros_like(x_sect_nondim)
    r_sect_nondim = np.ones_like(x_sect_nondim)

    x_fuse_sections.append(
        linear_map(
            f_in=x_sect_nondim,
            min_in=0, max_in=1,
            min_out=x_fwd_tank_to_cabin, max_out=x_cabin_to_aft_tank
        )
    )
    z_fuse_sections.append(
        linear_map(
            f_in=z_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )
    r_fuse_sections.append(
        linear_map(
            f_in=r_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )

    # Aft Tank
    x_sect_nondim = np.linspace(0, 1, 2)
    z_sect_nondim = np.zeros_like(x_sect_nondim)
    r_sect_nondim = np.ones_like(x_sect_nondim)

    x_fuse_sections.append(
        linear_map(
            f_in=x_sect_nondim,
            min_in=0, max_in=1,
            min_out=x_cabin_to_aft_tank, max_out=x_aft_tank_to_tail
        )
    )
    z_fuse_sections.append(
        linear_map(
            f_in=z_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )
    r_fuse_sections.append(
        linear_map(
            f_in=r_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )

    # Tail
    x_sect_nondim = np.linspace(0, 1, 10)
    z_sect_nondim = 1 * x_sect_nondim ** 1.5
    r_sect_nondim = 1 - x_sect_nondim ** 1.5

    x_fuse_sections.append(
        linear_map(
            f_in=x_sect_nondim,
            min_in=0, max_in=1,
            min_out=x_aft_tank_to_tail, max_out=x_tail
        )
    )
    z_fuse_sections.append(
        linear_map(
            f_in=z_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )
    r_fuse_sections.append(
        linear_map(
            f_in=r_sect_nondim,
            min_in=0, max_in=1,
            min_out=0, max_out=fuselage_cabin_radius
        )
    )

    # Compile Fuselage
    x_fuse_sections = np.concatenate([
        x_fuse_section[:-1] if i != len(x_fuse_sections) - 1 else x_fuse_section
        for i, x_fuse_section in enumerate(x_fuse_sections)
    ])
    z_fuse_sections = np.concatenate([
        z_fuse_section[:-1] if i != len(z_fuse_sections) - 1 else z_fuse_section
        for i, z_fuse_section in enumerate(z_fuse_sections)
    ])
    r_fuse_sections = np.concatenate([
        r_fuse_section[:-1] if i != len(r_fuse_sections) - 1 else r_fuse_section
        for i, r_fuse_section in enumerate(r_fuse_sections)
    ])

    fuse = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[
                    x_fuse_sections[i],
                    0,
                    z_fuse_sections[i]
                ],
                radius=r_fuse_sections[i]
            )
            for i in range(np.length(x_fuse_sections))
        ],
        analysis_specific_options={
            asb.AeroBuildup: dict(
                nose_fineness_ratio=nose_fineness_ratio
            )
        }
    )

    ### Wing
    wing_airfoil = asb.Airfoil("b737c").repanel(100)

    wing_span = 214 * u.foot
    wing_half_span = wing_span / 2

    wing_root_chord = 51.5 * u.foot

    wing_LE_sweep_deg = 34

    wing_yehudi_span_fraction = 0.25
    wing_dihedral = 6

    # Compute the y locations
    wing_yehudi_y = wing_yehudi_span_fraction * wing_half_span
    wing_tip_y = wing_half_span

    # Compute the x locations
    wing_yehudi_x = wing_yehudi_y * np.tand(wing_LE_sweep_deg)
    wing_tip_x = wing_tip_y * np.tand(wing_LE_sweep_deg)

    # Compute the chords
    wing_yehudi_chord = wing_root_chord - wing_yehudi_x
    wing_tip_chord = 0.14 * wing_root_chord

    # Make the sections
    wing_root = asb.WingXSec(
        xyz_le=[0, 0, 0],
        chord=wing_root_chord,
        airfoil=wing_airfoil,
    )
    wing_yehudi = asb.WingXSec(
        xyz_le=[
            wing_yehudi_x,
            wing_yehudi_y,
            wing_yehudi_y * np.tand(wing_dihedral)
        ],
        chord=wing_yehudi_chord,
        airfoil=wing_airfoil,
    )
    wing_tip = asb.WingXSec(
        xyz_le=[
            wing_tip_x,
            wing_tip_y,
            wing_tip_y * np.tand(wing_dihedral)
        ],
        chord=wing_tip_chord,
        airfoil=wing_airfoil
    )

    # Assemble the wing
    wing_x_le = 0.5 * x_fwd_tank_to_cabin + 0.5 * x_cabin_to_aft_tank - 0.5 * wing_root_chord

    wing_z_le = -0.5 * fuselage_cabin_radius

    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            wing_root,
            wing_yehudi,
            wing_tip
        ]
    ).translate([
        wing_x_le,
        0,
        wing_z_le
    ]).subdivide_sections(3)

    ### Horizontal Stabilizer
    hstab_span = 70.8 * u.foot
    hstab_half_span = hstab_span / 2

    hstab_root_chord = 23 * u.foot

    hstab_LE_sweep_deg = 39

    hstab_root = asb.WingXSec(
        xyz_le=[0, 0, 0],
        chord=hstab_root_chord,
        airfoil=asb.Airfoil("naca0012"),
        control_surfaces=[
            asb.ControlSurface(
                name="All-moving Elevator",
                deflection=0
            )
        ]
    )
    hstab_tip = asb.WingXSec(
        xyz_le=[
            hstab_half_span * np.tand(hstab_LE_sweep_deg),
            hstab_half_span,
            0
        ],
        chord=0.35 * hstab_root_chord,
        airfoil=asb.Airfoil("naca0012")
    )

    # Assemble the hstab
    hstab_x_le = x_tail - 2 * hstab_root_chord
    hstab_z_le = 0.5 * fuselage_cabin_radius

    hstab = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            hstab_root,
            hstab_tip
        ]
    ).translate([
        hstab_x_le,
        0,
        hstab_z_le
    ]).subdivide_sections(3)

    ### Vertical Stabilizer
    vstab_span = 29.6 * u.foot

    vstab_root_chord = 22 * u.foot

    vstab_LE_sweep_deg = 45

    vstab_root = asb.WingXSec(
        xyz_le=[0, 0, 0],
        chord=vstab_root_chord,
        airfoil=asb.Airfoil("naca0008")
    )
    vstab_tip = asb.WingXSec(
        xyz_le=[
            vstab_span * np.tand(vstab_LE_sweep_deg),
            0,
            vstab_span,
        ],
        chord=0.35 * vstab_root_chord,
        airfoil=asb.Airfoil("naca0008")
    )

    # Assemble the vstab
    vstab_x_le = x_tail - 2 * vstab_root_chord
    vstab_z_le = 1 * fuselage_cabin_radius

    vstab = asb.Wing(
        name="Vertical Stabilizer",
        xsecs=[
            vstab_root,
            vstab_tip
        ]
    ).translate([
        vstab_x_le,
        0,
        vstab_z_le
    ]).subdivide_sections(3)

    ### Airplane
    airplane = asb.Airplane(
        name="Airplane",
        xyz_ref=[],
        wings=[
            wing,
            hstab,
            vstab
        ],
        fuselages=[
            fuse
        ],
    )

    return airplane


if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    airplane = a()

    airplane.draw_three_view()

    # airplane.draw(backend="matplotlib", show=False)
    #
    # import matplotlib.pyplot as plt
    # import aerosandbox.tools.pretty_plots as p
    #
    # ax = plt.gca()
    #
    # t = np.linspace(0, 1, 100)
    #
    # x = airplane.fuselages[0].length() * t
    # y = 15 * np.sin(4 * 2 * np.pi * t)
    # z = 15 *np.cos(4 * 2 * np.pi * t)
    #
    # ax.plot(
    #     x, y, z
    # )
    #
    # p.show_plot()
