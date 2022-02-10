import aerosandbox.numpy as np


def peak_of_harmonic_oscillation(
        amplitude: float = 1,
        frequency_hz: float = 1,
        derivative_order: int = 0,
):
    """
    Computes the peak value of the nth derivative of a simple harmonic oscillation (e.g., sine wave).

    Specifically, if `t` represents time, and we have an oscillation that is described by a function f(t) as:

        `f(t) = amplitude * sin(2 * pi * t * frequency_hz)`

    then this function will return the value of the nth time derivative of f(t), where `derivative_order` gives the
    value of n.

    Example:

        My spring-mass system has an oscillating position, where the mass oscillates in position from -0.1 to 0.1,
        with a frequency of 3 Hz. I want to know what the peak acceleration on that mass is:

        >>> peak_of_harmonic_oscillation(
        >>>     amplitude=0.1,
        >>>     frequency_hz=3,
        >>>     derivative_order=2
        >>> )

        This would return (2 * pi * 3) ^ 2 * 0.1, or around 35.5.

        So, the peak acceleration on my mass is around 35.5 m/s^2.

    Args:

        amplitude: Amplitude of the underlying oscillation. Amplitude here is the center-to-peak distance,
        not peak-to-peak.

        frequency_hz: Frequency of the underlying oscillation, in units of 1/time (Hz). Note: not the angular frequency.

        derivative_order: The derivative order. (E.g., 0 for position, 1 for velocity, 2 for acceleration).

    Returns: The peak value of the nth derivative of a simple harmonic oscillator with the specified amplitude and frequency.

    """
    return (2 * np.pi * frequency_hz) ** derivative_order * amplitude
