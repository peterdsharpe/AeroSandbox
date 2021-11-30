import aerosandbox.numpy as np


def power_human(
        duration,  # type: float
        dataset="Healthy Men"  # type: str
):
    """
    Finds the power output that a human can sustain for a given duration.
    Data was fit for durations in the range of 6 seconds to 60,000 seconds.
    Fits are modeled at: AeroSandbox/studies/HumanPower
    Data Source: Bicycling Science by D. Wilson, 2004. Figure 2.4.
        Wilson is aggregating many data sources here.
        The raw data pulls from a variety of sources:
            * NASA SP-3006, 1964
            * U.K. amateur trials and time-trials records (Whitt, F.R. 1971 "A note on the estimation of the energy expenditure of sporting cyclists." Ergonomics 14)
            * Wilsons' own analyses
    Weight estimates for tests subjects are unfortunately not given.
    :param duration: Time to sustain power output [seconds]
    :param dataset: Dataset to pull from. A string that is one of the following:
        "Healthy Men",
        "First-Class Athletes",
        "World-Class Athletes",
    :return: Sustainable power output for the specified duration [W]
    """
    if dataset == "Healthy Men":
        a = 373.153360
        b0 = -0.173127
        b1 = 0.083282
        b2 = -0.042785
    elif dataset == "First-Class Athletes":
        a = 502.332185
        b0 = -0.179030
        b1 = 0.097926
        b2 = -0.024855
    elif dataset == "World-Class Athletes":
        a = 869.963370
        b0 = -0.234291
        b1 = 0.064395
        b2 = -0.009197
    else:
        raise ValueError("Bad value of 'dataset'!")

    duration_mins = duration / 60
    log_duration_mins = np.log10(duration_mins)

    return a * duration_mins ** (
            b0 +
            b1 * log_duration_mins +
            b2 * log_duration_mins ** 2
    ) # essentially, a cubic in log-log space

if __name__ == '__main__':
    print(
        power_human(
            duration=60,
        )
    )