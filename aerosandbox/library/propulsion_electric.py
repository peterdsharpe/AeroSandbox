import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from typing import Union, Dict


def motor_electric_performance(
        voltage: Union[float, np.ndarray] = None,
        current: Union[float, np.ndarray] = None,
        rpm: Union[float, np.ndarray] = None,
        torque: Union[float, np.ndarray] = None,
        kv: float = 1000.,  # rpm/volt
        resistance: float = 0.1,  # ohms
        no_load_current: float = 0.4  # amps
) -> Dict[str, Union[float, np.ndarray]]:
    """
    A function for predicting the performance of an electric motor.
    
    Performance equations based on Mark Drela's First Order Motor Model:
    http://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf
    
    Instructions: Input EXACTLY TWO of the following parameters: voltage, current, rpm, torque.
    
    Exception: You cannot supply the combination of current and torque - this makes for an ill-posed problem.
    
    Note that this function is fully vectorized, so arrays can be supplied to any of the inputs.
    
    Args:
        voltage: Voltage across motor terminals [Volts]
        
        current: Current through motor [Amps]
        
        rpm: Motor rotation speed [rpm]
        
        torque: Motor torque [N m]
        
        kv: voltage constant, in rpm/volt
        
        resistance: resistance, in ohms
        
        no_load_current: no-load current, in amps
        
    Returns:
        A dictionary where keys are: 
            "voltage", 
            "current", 
            "rpm", 
            "torque",
            "shaft power", 
            "electrical power", 
            "efficiency"
            "waste heat"

        And values are corresponding quantities in SI units.

        Note that "efficiency" is just (shaft power) / (electrical power), and hence implicitly assumes that the
        motor is operating as a motor (electrical -> shaft power), and not a generator (shaft power -> electrical).
        If you want to know the efficiency of the motor as a generator, you can simply calculate it as (electrical
        power) / (shaft power).
    """
    # Validate inputs
    voltage_known = voltage is not None
    current_known = current is not None
    rpm_known = rpm is not None
    torque_known = torque is not None

    if not (
            voltage_known + current_known + rpm_known + torque_known == 2
    ):
        raise ValueError("You must give exactly two input arguments.")

    if current_known and torque_known:
        raise ValueError(
            "You cannot supply the combination of current and torque - this makes for an ill-posed problem.")

    kv_rads_per_sec_per_volt = kv * np.pi / 30  # rads/sec/volt

    ### Iterate through the motor equations until all quantities are known.
    while not (voltage_known and current_known and rpm_known and torque_known):
        if rpm_known:
            if current_known and not voltage_known:
                speed = rpm * np.pi / 30  # rad/sec
                back_EMF_voltage = speed / kv_rads_per_sec_per_volt
                voltage = back_EMF_voltage + current * resistance
                voltage_known = True

        if torque_known:
            if not current_known:
                current = torque * kv_rads_per_sec_per_volt + no_load_current
                current_known = True

        if voltage_known:
            if rpm_known and not current_known:
                speed = rpm * np.pi / 30  # rad/sec
                back_EMF_voltage = speed / kv_rads_per_sec_per_volt
                current = (voltage - back_EMF_voltage) / resistance
                current_known = True
            if not rpm_known and current_known:
                back_EMF_voltage = voltage - (current * resistance)
                speed = back_EMF_voltage * kv_rads_per_sec_per_volt
                rpm = speed * 30 / np.pi
                rpm_known = True

        if current_known:
            if not torque_known:
                torque = (current - no_load_current) / kv_rads_per_sec_per_volt
                torque_known = True

    shaft_power = (rpm * np.pi / 30) * torque
    electrical_power = voltage * current
    efficiency = shaft_power / electrical_power
    waste_heat = np.fabs(electrical_power - shaft_power)

    return {
        "voltage"         : voltage,
        "current"         : current,
        "rpm"             : rpm,
        "torque"          : torque,
        "shaft power"     : shaft_power,
        "electrical power": electrical_power,
        "efficiency"      : efficiency,
        "waste heat"      : waste_heat,
    }


def motor_resistance_from_no_load_current(
        no_load_current
):
    """
    Estimates the internal resistance of a motor from its no_load_current. Gates quotes R^2=0.93 for this model.

    Source: Gates, et. al., "Combined Trajectory, Propulsion, and Battery Mass Optimization for Solar-Regen..."
        https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=3932&context=facpub

    Args:
        no_load_current: No-load current [amps]

    Returns:
        motor internal resistance [ohms]
    """
    return 0.0467 * no_load_current ** -1.892


def mass_ESC(
        max_power,
):
    """
    Estimates the mass of an ESC.

    Informal correlation I did to Hobbyking ESCs in the 8S LiPo, 100A range

    Args:
        max_power: maximum power [W]

    Returns:
        estimated ESC mass [kg]
    """
    return 2.38e-5 * max_power


def mass_battery_pack(
        battery_capacity_Wh,
        battery_cell_specific_energy_Wh_kg=240,
        battery_pack_cell_fraction=0.7,
):
    """
    Estimates the mass of a lithium-polymer battery.

    Args:
        battery_capacity_Wh: Battery capacity, in Watt-hours [W*h]

        battery_cell_specific_energy: Specific energy of the battery at the CELL level [W*h/kg]

        battery_pack_cell_fraction: Fraction of the battery pack that is cells, by weight.

            * Note: Ed Lovelace, a battery engineer for Aurora Flight Sciences, gives this figure as 0.70 in a Feb.
            2020 presentation for MIT 16.82

    Returns:
        Estimated battery mass [kg]
    """
    return battery_capacity_Wh / battery_cell_specific_energy_Wh_kg / battery_pack_cell_fraction


def mass_motor_electric(
        max_power,
        kv_rpm_volt=1000,  # This is in rpm/volt, not rads/sec/volt!
        voltage=20,
        method="hobbyking"
):
    """
    Estimates the mass of a brushless DC electric motor.
    Curve fit to scraped Hobbyking BLDC motor data as of 2/24/2020.
    Estimated range of validity: 50 < max_power < 10000

    Args:
        max_power (float): maximum power [W]

        kv_rpm_volt (float): Voltage constant of the motor, measured in rpm/volt, not rads/sec/volt! [rpm/volt]

        voltage (float): Operating voltage of the motor [V]

        method (str): method to use. "burton", "hobbyking", or "astroflight" (increasing level of detail).

            * Burton source: https://dspace.mit.edu/handle/1721.1/112414

            * Hobbyking source: C:\Projects\GitHub\MotorScraper,

            * Astroflight source: Gates, et. al., "Combined Trajectory, Propulsion, and Battery Mass Optimization for
            Solar-Regen..." https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=3932&context=facpub

                * Validity claimed from 1.5 kW to 15 kW, kv from 32 to 1355.

    Returns:
        Estimated motor mass [kg]
    """
    if method == "burton":
        return max_power / 4128  # Less sophisticated model. 95% CI (3992, 4263), R^2 = 0.866
    elif method == "hobbyking":
        return 10 ** (0.8205 * np.log10(max_power) - 3.155)  # More sophisticated model
    elif method == "astroflight":
        max_current = max_power / voltage
        return 2.464 * max_current / kv_rpm_volt + 0.368  # Even more sophisticated model


def mass_wires(
        wire_length,
        max_current,
        allowable_voltage_drop,
        material="aluminum",
        insulated=True,
        max_voltage=600,
        wire_packing_factor=1,
        insulator_density=1700,
        insulator_dielectric_strength=12e6,
        insulator_min_thickness=0.2e-3,  # silicone wire
        return_dict: bool = False
):
    """
    Estimates the mass of wires used for power transmission.

    Materials data from: https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity#Resistivity-density_product
        All data measured at STP; beware, as this data (especially resistivity) can be a strong function of temperature.

    Args:
        wire_length (float): Length of the wire [m]

        max_current (float): Max current of the wire [Amps]

        allowable_voltage_drop (float): How much is the voltage allowed to drop along the wire?

        material (str): Conductive material of the wire ("aluminum"). Determines density and resistivity. One of:

            * "sodium"

            * "lithium"

            * "calcium"

            * "potassium"

            * "beryllium"

            * "aluminum"

            * "magnesium"

            * "copper"

            * "silver"

            * "gold"

            * "iron"

        insulated (bool): Should we add the mass of the wire's insulator coating? Usually you'll want to leave this True.

        max_voltage (float): Maximum allowable voltage (used for sizing insulator). 600 is a common off-the-shelf rating.

        wire_packing_factor (float): What fraction of the enclosed cross section is conductor? This is 1 for solid wire,
            and less for stranded wire.

        insulator_density (float): Density of the insulator [kg/m^3]

        insulator_dielectric_strength (float): Dielectric strength of the insulator [V/m]. The default value of 12e6 corresponds
        to rubber.

        insulator_min_thickness (float): Minimum thickness of the insulator [m]. This is essentially a gauge limit.
        The default value is 0.2 mm.

        return_dict (bool): If True, returns a dictionary of all local variables. If False, just returns the wire
        mass as a float. Defaults to False.


    Returns: If `return_dict` is False (default), returns the wire mass as a single number. If `return_dict` is True,
    returns a dictionary of all local variables.
    """
    if material == "sodium":  # highly reactive with water & oxygen, low physical strength
        density = 970  # kg/m^3
        resistivity = 47.7e-9  # ohm-meters
    elif material == "lithium":  # highly reactive with water & oxygen, low physical strength
        density = 530  # kg/m^3
        resistivity = 92.8e-9  # ohm-meters
    elif material == "calcium":  # highly reactive with water & oxygen, low physical strength
        density = 1550  # kg/m^3
        resistivity = 33.6e-9  # ohm-meters
    elif material == "potassium":  # highly reactive with water & oxygen, low physical strength
        density = 890  # kg/m^3
        resistivity = 72.0e-9  # ohm-meters
    elif material == "beryllium":  # toxic, brittle
        density = 1850  # kg/m^3
        resistivity = 35.6e-9  # ohm-meters
    elif material == "aluminum":
        density = 2700  # kg/m^3
        resistivity = 26.50e-9  # ohm-meters
    elif material == "magnesium":  # worse specific conductivity than aluminum
        density = 1740  # kg/m^3
        resistivity = 43.90e-9  # ohm-meters
    elif material == "copper":  # worse specific conductivity than aluminum, moderately expensive
        density = 8960  # kg/m^3
        resistivity = 16.78e-9  # ohm-meters
    elif material == "silver":  # worse specific conductivity than aluminum, expensive
        density = 10490  # kg/m^3
        resistivity = 15.87e-9  # ohm-meters
    elif material == "gold":  # worse specific conductivity than aluminum, very expensive
        density = 19300  # kg/m^3
        resistivity = 22.14e-9  # ohm-meters
    elif material == "iron":  # worse specific conductivity than aluminum
        density = 7874  # kg/m^3
        resistivity = 96.1e-9  # ohm-meters
    else:
        raise ValueError("Bad value of 'material'!")

    # Conductor mass
    resistance = allowable_voltage_drop / max_current
    area_conductor = resistivity * wire_length / resistance
    volume_conductor = area_conductor * wire_length
    mass_conductor = volume_conductor * density

    # Insulator mass
    if insulated:
        insulator_thickness = np.softmax(
            4.0 * max_voltage / insulator_dielectric_strength,
            insulator_min_thickness,
            softness=0.005 * u.inch,
        )
        radius_conductor = (area_conductor / wire_packing_factor / np.pi) ** 0.5
        radius_insulator = radius_conductor + insulator_thickness
        area_insulator = np.pi * radius_insulator ** 2 - area_conductor
        volume_insulator = area_insulator * wire_length
        mass_insulator = insulator_density * volume_insulator
    else:
        mass_insulator = 0

    # Total them up
    mass_total = mass_conductor + mass_insulator

    if return_dict:
        return locals()
    else:
        return mass_total


if __name__ == '__main__':
    print(motor_electric_performance(
        rpm=100,
        current=3
    ))
    print(motor_electric_performance(
        rpm=4700,
        torque=0.02482817
    ))

    print(
        mass_battery_pack(100)
    )

    pows = np.logspace(2, 5, 300)
    mass_mot_burton = mass_motor_electric(pows, method="burton")
    mass_mot_hobbyking = mass_motor_electric(pows, method="hobbyking")
    mass_mot_astroflight = mass_motor_electric(pows, method="astroflight")

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.loglog(pows, np.array(mass_mot_burton), "-", label="Burton Model")
    plt.plot(pows, np.array(mass_mot_hobbyking), "--", label="Hobbyking Model")
    plt.plot(pows, np.array(mass_mot_astroflight), "-.", label="Astroflight Model")
    p.show_plot(
        "Small Electric Motor Mass Models\n(500 kv, 100 V)",
        "Motor Power [W]",
        "Motor Mass [kg]"
    )

    print(mass_wires(
        wire_length=1,
        max_current=100,
        allowable_voltage_drop=1,
        material="aluminum"
    ))
