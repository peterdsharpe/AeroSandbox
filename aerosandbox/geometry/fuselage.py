from aerosandbox import AeroSandboxObject
from aerosandbox.geometry.common import *
from typing import List


class Fuselage(AeroSandboxObject):
    """
    Definition for a fuselage or other slender body (pod, etc.).
    For now, all fuselages are assumed to be circular and fairly closely aligned with the body x axis. (<10 deg or so) # TODO update if this changes
    """

    def __init__(self,
                 name: str = "Untitled Fuselage",  # It can help when debugging to give each fuselage a sensible name.
                 x_le: float = 0,
                 # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 y_le: float = 0,
                 # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 z_le: float = 0,
                 # Will translate all of the xsecs of the fuselage. Useful for moving the fuselage around.
                 xsecs: List['FuselageXSec'] = [],  # This should be a list of FuselageXSec objects.
                 symmetric: bool = False,  # Is the fuselage symmetric across the XZ plane?
                 circumferential_panels: int = 24,
                 # Number of circumferential panels to use in VLM and Panel analysis. Should be even.
                 ):
        self.name = name
        self.xyz_le = np.array([x_le, y_le, z_le])
        self.xsecs = xsecs
        self.symmetric = symmetric
        if not circumferential_panels % 2 == 0:
            raise ValueError("You should use an even number of circumferential panels to avoid symmetry problems.")
        self.circumferential_panels = circumferential_panels

    def area_wetted(self) -> float:
        """
        Returns the wetted area of the fuselage.

        If the Fuselage is symmetric (i.e. two symmetric wingtip pods),
        returns the combined wetted area of both pods.
        :return:
        """
        area = 0
        for i in range(len(self.xsecs) - 1):
            this_radius = self.xsecs[i].radius
            next_radius = self.xsecs[i + 1].radius
            x_separation = self.xsecs[i + 1].x_c - self.xsecs[i].x_c
            area += np.pi * (this_radius + next_radius) * np.sqrt(
                (this_radius - next_radius) ** 2 + x_separation ** 2)
        if self.symmetric:
            area *= 2
        return area

    #
    def area_projected(self) -> float:
        """
        Returns the area of the fuselage as projected onto the XY plane (top-down view).

        If the Fuselage is symmetric (i.e. two symmetric wingtip pods),
        returns the combined projected area of both pods.
        :return:
        """
        area = 0
        for i in range(len(self.xsecs) - 1):
            this_radius = self.xsecs[i].radius
            next_radius = self.xsecs[i + 1].radius
            x_separation = self.xsecs[i + 1].x_c - self.xsecs[i].x_c
            area += (this_radius + next_radius) * x_separation
        if self.symmetric:
            area *= 2
        return area

    def length(self) -> float:
        """
        Returns the total front-to-back length of the fuselage. Measured as the difference between the x-coordinates
        of the leading and trailing cross sections.
        :return:
        """
        return np.fabs(self.xsecs[-1].x_c - self.xsecs[0].x_c)

    def volume(self) -> float:
        """
        Gives the volume of the Fuselage.

        Returns:
            Fuselage volume.
        """
        volume = 0
        for i in range(len(self.xsecs) - 1):
            xsec_a, xsec_b = self.xsecs[i], self.xsecs[i + 1]
            h = np.abs(xsec_b.x_c - xsec_a.x_c)
            radius_a = xsec_a.radius
            radius_b = xsec_b.radius
            volume += np.pi * h / 3 * (
                    radius_a ** 2 + radius_a * radius_b + radius_b ** 2
            )
        return volume


class FuselageXSec(AeroSandboxObject):
    """
    Definition for a fuselage cross section ("X-section").
    """

    def __init__(self,
                 x_c=0,
                 y_c=0,
                 z_c=0,
                 radius=0,
                 ):
        self.x_c = x_c
        self.y_c = y_c
        self.z_c = z_c

        self.radius = radius

        self.xyz_c = np.array([x_c, y_c, z_c])


    def xsec_area(self):
        """
        Returns the FuselageXSec's cross-sectional (xsec) area.
        :return:
        """
        return np.pi * self.radius ** 2
