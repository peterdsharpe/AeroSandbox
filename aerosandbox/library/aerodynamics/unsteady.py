import matplotlib.pyplot as plt
import aerosandbox.numpy as np
from typing import Union,List

def wagners_function(reduced_time: Union[List,float]):
    """ 
    A commonly used approximation to Wagner's function 
    (Jones, R.T. The Unsteady Lift of a Finite Wing; Technical Report NACA TN-682; NACA: Washington, DC, USA, 1939)
    
    Args:
        reduced_time (float,List) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        """
    return (1 - 0.165 * np.exp(-0.0455 * reduced_time) - 
                0.335 * np.exp(-0.3 * reduced_time))


def calculate_wagner_lift_coefficient(
        reduced_time: Union[float,List] , 
        angle_of_attack: float
):
    """
    Computes the evolution of the lift coefficient in Wagner's problem which can be interpreted as follows
    1) An impulsively started thin airfoil at consntant angle of attack
    2) An impuslive change in the angle of attack of the the thin airfoil at constant velocity
   
    The model predicts infinite added mass at the first instant due to the infinite acceleration
    The delta function term (and therefore added mass) has been ommited in this case.
    The way the function is set up right now, reduced_time = 0 corresponds to the instance the airfoil pitches/accelerates

    """
    return 2 * np.pi * angle_of_attack * wagners_function(reduced_time)


def calculate_kussner_lift_coefficient(
        reduced_time: Union[float,List] , 
        gust_strength:float ,
        velocity: float
):
    """
    Computes the evolution of the lift coefficient of a thin airfoil entering a sharp, transverse, top hat gust
    The way the function is set up right now, reduced_time = 0 corresponds to the instance the gust is entered
    
    (Leishman, Principles of Helicopter Aerodynamics, S8.10,S8.11)
    
    Args:
        reduced_time (float,List) : Reduced time, equal to the number of semichords travelled. See function reduced_time
        gust_strength (float) : velocity in m/s of the top hat gust
        velocity (float) : velocity of the thin airfoil entering the gust

    """
    return 2 * np.pi * np.arctan(gust_strength / velocity) * kussners_function(reduced_time) 




def kussners_function(reduced_time: Union[List,float]):
    """ 
    A commonly used approximation to Kussner's function (Sears and Sparks 1941)
    
    Args:
        reduced_time (float,List) : Reduced time, equal to the number of semichords travelled. See function reduced_time
    """
    return 1 - 0.5 * np.exp(-0.13 * reduced_time) - 0.5 * np.exp(-reduced_time) 

def calculate_reduced_time(
        time: Union[float,List],
        velocity: Union[float,List],
        chord: float 
) -> Union[float,List]:
    """ 
    Calculates reduced time from time in seconds and velocity history in m/s. For constant velocity it reduces to s = 2*U*t/c
    The reduced time is the number of semichords travelled by the airfoil/aircaft i.e. 2 / chord * integral from t0 to t of velocity dt  
    
    
    Args:
        time (float,List) : Time in seconds 
        velocity (float,List) : Either constant velocity or array of velocities at corresponding times
        chord (float) : The chord of the airfoil
        
    Returns:
        The reduced time as an array or similar to the input. The first element is 0. 
    """
    
    if type(velocity) == float or type(velocity) == int : 
        return 2 * velocity * time / chord
    else:
        assert np.size(velocity) == np.size(time)
        reduced_time = np.zeros_like(time)
        for i in range(len(time)-1):
            reduced_time[i+1] =  reduced_time[i] + (velocity[i+1] + velocity[i])/2 * (time[i+1]-time[i])        
        return 2 / chord * reduced_time
    
    