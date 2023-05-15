import aerosandbox as asb
import aerosandbox.numpy as np
import cadquery as cq
import copy
from typing import List, Union, Dict
from sortedcontainers import SortedDict


class WingStructureGenerator():
    def __init__(self,
                 wing: asb.Wing,
                 default_rib_thickness=3e-3,
                 minimum_airfoil_TE_thickness_rel: float = 0.001
                 ):
        self.wing = wing
        self.default_rib_thickness = default_rib_thickness
        self.minimum_airfoil_TE_thickness_rel = minimum_airfoil_TE_thickness_rel

        ### Compute some span properties which are used for locating ribs
        self._sectional_spans: List[float] = wing.span(_sectional=True)
        self._cumulative_spans_up_to_section = np.concatenate((
            [0],
            np.cumsum(self._sectional_spans)
        ))
        self._total_span = sum(self._sectional_spans)

        ### Generate the OML geometry
        self.oml = asb.Airplane(
            wings=[wing]
        ).generate_cadquery_geometry(
            minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness_rel
        )

        ### Set up data structures for geometry
        self.ribs: SortedDict[cq.Workplane] = SortedDict()
        self.spars: List[cq.Workplane] = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.wing})"

    # def open_interactive(self):

    def add_ribs_from_section_span_fractions(
            self,
            section_index: int,
            section_span_fractions: Union[float, int, List[float], np.ndarray],
            rib_thickness: float = None,
    ):
        if rib_thickness is None:
            rib_thickness = self.default_rib_thickness

        try:
            iter(section_span_fractions)
        except TypeError:
            section_span_fractions = [section_span_fractions]

        xsec_a = wing.xsecs[section_index]
        xsec_b = wing.xsecs[section_index + 1]

        for s in section_span_fractions:
            af = xsec_a.airfoil.blend_with_another_airfoil(
                airfoil=xsec_b.airfoil,
                blend_fraction=s
            )

            chord = (
                    (1 - s) * xsec_a.chord + s * xsec_b.chord
            )

            csys = wing._compute_frame_of_section(section_index)

            span = (
                    self._cumulative_spans_up_to_section[section_index]
                    + s * self._sectional_spans[section_index]
            )

            self.ribs[span] = (
                cq.Workplane(
                    inPlane=cq.Plane(
                        origin=tuple(
                            (1 - s) * xsec_a.xyz_le + s * xsec_b.xyz_le
                        ),
                        xDir=tuple(csys[0]),
                        normal=tuple(-csys[1])
                    )
                ).spline(
                    listOfXYTuple=[
                        tuple(xy * chord)
                        for xy in af.coordinates
                    ]
                ).close().extrude(
                    rib_thickness / 2,
                    combine=False,
                    both=True
                )
            )

    def add_ribs_from_xsecs(
            self,
            indexes: List[int] = None,
            rib_thickness: float = None
    ):
        if rib_thickness is None:
            rib_thickness = self.default_rib_thickness

        if indexes is None:
            indexes = range(len(wing.xsecs))

        for i in indexes:
            xsec = wing.xsecs[i]
            csys = wing._compute_frame_of_WingXSec(i)

            af = xsec.airfoil
            if af.TE_thickness() < self.minimum_airfoil_TE_thickness_rel:
                af = af.set_TE_thickness(
                    thickness=self.minimum_airfoil_TE_thickness_rel
                )

            span = self._cumulative_spans_up_to_section[i]

            self.ribs[span] = (
                cq.Workplane(
                    inPlane=cq.Plane(
                        origin=tuple(xsec.xyz_le),
                        xDir=tuple(csys[0]),
                        normal=tuple(-csys[1])
                    )
                ).spline(
                    listOfXYTuple=[
                        tuple(xy * xsec.chord)
                        for xy in af.coordinates
                    ]
                ).close().extrude(
                    rib_thickness / 2,
                    combine=False,
                    both=True
                )
            )

    def add_ribs_from_span_fractions(
            self,
            span_fractions: Union[float, List[float], np.ndarray] = np.linspace(0, 1, 10),
            rib_thickness: float = None,
    ):
        ### Handle span_fractions if it's not an iterable
        try:
            iter(span_fractions)
        except TypeError:
            span_fractions = [span_fractions]

        for s in span_fractions:
            if s == 0:
                section_index = 0
                section_span_fraction = 0
            elif s == 1:
                section_index = len(self.wing.xsecs) - 2
                section_span_fraction = 1
            elif s < 0 or s > 1:
                raise ValueError(
                    "All values of `span_fractions` must be between 0 and 1!"
                )
            else:
                section_index = np.argwhere(
                    self._cumulative_spans_up_to_section > self._total_span * s
                )[0][0] - 1

                section_span_fraction = (
                                                s * self._total_span
                                                - self._cumulative_spans_up_to_section[section_index]
                                        ) / self._sectional_spans[section_index]

            self.add_ribs_from_section_span_fractions(
                section_index=section_index,
                section_span_fractions=section_span_fraction,
                rib_thickness=rib_thickness
            )

    def add_tube_spar(self,
                      span_location_root: float,
                      span_location_tip: float,
                      diameter_root,
                      x_over_c_location_root=0.25,
                      y_over_c_location_root=None,
                      x_over_c_location_tip=None,
                      y_over_c_location_tip=None,
                      diameter_tip: float = None,
                      cut_ribs: bool = True,
                      ):
        if diameter_tip is None:
            diameter_tip = diameter_root
        if x_over_c_location_tip is None:
            x_over_c_location_tip = x_over_c_location_root
            # TODO change behavior so that default is a 90 degree spar
        if y_over_c_location_root is None:
            y_over_c_location_root = wing.xsecs[0].airfoil.local_camber(
                x_over_c=x_over_c_location_root
            )
        if y_over_c_location_tip is None:
            y_over_c_location_tip = y_over_c_location_root
            # TODO change behavior so that default is a 90 degree spar

        ### Figure out where the spar root is
        section_index = np.argwhere(
            self._cumulative_spans_up_to_section > span_location_root
        )[0][0] - 1

        section_span_fraction = (
                                        span_location_root
                                        - self._cumulative_spans_up_to_section[section_index]
                                ) / self._sectional_spans[section_index]

        root_csys = self.wing._compute_frame_of_section(section_index)

        root_le_point = (
                (1 - section_span_fraction) * wing.xsecs[section_index].xyz_le
                + section_span_fraction * wing.xsecs[section_index + 1].xyz_le
        )
        root_chord = (
                (1 - section_span_fraction) * wing.xsecs[section_index].chord
                + section_span_fraction * wing.xsecs[section_index + 1].chord
        )
        root_point = (
                root_le_point +
                x_over_c_location_root * root_csys[0] * root_chord +
                y_over_c_location_root * root_csys[2] * root_chord
        )

        ### Figure out where the spar tip is

        section_index = np.argwhere(
            self._cumulative_spans_up_to_section > span_location_tip
        )[0][0] - 1

        section_span_fraction = (
                                        span_location_tip
                                        - self._cumulative_spans_up_to_section[section_index]
                                ) / self._sectional_spans[section_index]

        tip_csys = self.wing._compute_frame_of_section(section_index)

        tip_le_point = (
                (1 - section_span_fraction) * wing.xsecs[section_index].xyz_le
                + section_span_fraction * wing.xsecs[section_index + 1].xyz_le
        )
        tip_chord = (
                (1 - section_span_fraction) * wing.xsecs[section_index].chord
                + section_span_fraction * wing.xsecs[section_index + 1].chord
        )
        tip_point = (
                tip_le_point +
                x_over_c_location_tip * tip_csys[0] * tip_chord +
                y_over_c_location_tip * tip_csys[2] * tip_chord
        )

        normal = tip_point - root_point

        root_plane = cq.Plane(
            origin=tuple(root_point),
            xDir=tuple(root_csys[0]),
            normal=tuple(normal)
        )

        tip_plane = cq.Plane(
            origin=tuple(tip_point),
            xDir=tuple(tip_csys[0]),
            normal=tuple(normal)
        )

        ### Make the spar
        root_wire = cq.Workplane(
            inPlane=root_plane
        ).circle(radius=2 * diameter_root / 2)
        tip_wire = cq.Workplane(
            inPlane=tip_plane
        ).circle(radius=2 * diameter_tip / 2)

        wire_collection = root_wire
        wire_collection.ctx.pendingWires.extend(tip_wire.ctx.pendingWires)

        spar = wire_collection.loft(ruled=True, clean=False)

        self.spars.append(spar)

        if cut_ribs:
            for span_loc, rib in self.ribs.items():
                rib: cq.Workplane
                self.ribs[span_loc] = rib.cut(spar)


from aerosandbox.tools import units as u

af = asb.Airfoil("sd7032")

wing = asb.Wing(
    name="Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            chord=0.2,
            airfoil=af
        ),
        asb.WingXSec(
            xyz_le=[0.05, 0.5, 0],
            chord=0.15,
            airfoil=af
        ),
        asb.WingXSec(
            xyz_le=[0.1, 0.8, 0.1],
            chord=0.1,
            airfoil=af
        )
    ]
)

s = WingStructureGenerator(
    wing,
    default_rib_thickness=1 / 16 * u.inch
)
# s.add_ribs_from_section_span_fractions(0, np.linspace(0, 1, 10)[1:-1])
# s.add_ribs_from_section_span_fractions(1, np.linspace(0, 1, 8)[1:-1])
s.add_ribs_from_xsecs()
# s.add_ribs_from_span_fractions(
#     span_fractions=np.linspace(0, 1, 6)
# )

s.add_tube_spar(
    diameter_root=8e-3,
    span_location_root=-1 / 16 * u.inch,
    span_location_tip=s.ribs.keys()[1],
    diameter_tip=4e-3
)

### Show all ribs
rs = s.ribs.values()[0]
for rib in s.ribs.values()[1:]:
    rs = rs.union(rib)

ss = s.spars[0]

# oml = s.oml
