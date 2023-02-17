# Control Surface Effectiveness

When doing simplified 2D aerodynamic modeling of airfoils with control surface deflections, it's often convenient to be able to convert a control surface deflection into a change in the effective angle of attack of the undeflected airfoil.

Of course, this conversion will depend (somewhat nonlinearly) on the hinge point, and probably also with Reynolds number, airfoil thickness, etc.

The only literature I can find that presents this is in an aircraft design textbook from Sadraey that shall not be named. It's reproduced here below, but **do not use this chart** (axis labels cut of deliberately to ensure this, but it's chord fraction vs. effectiveness).

![badplot](assets/do-not-use-this-plot.png)

The reason you should not use this plot is because this textbook from Sadraey is hot garbage - there is material that is totally inconsistent with theory and confidently incorrect. In the words of Mark Drela, "That Sadraey textbook is the aerodynamics equivalent of astrology."

So, we endeavour here to do a bit better by recreating this relationship.

