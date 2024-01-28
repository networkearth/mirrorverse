"""
Metabolic Inputs

.. admonition:: Core Idea

    In this model we are concerned with roughly 3 things:
        - :math:`r_{prey}`: rate of prey encounter (grams of prey per hour per square centimeter)
        - :math:`P_{att}`: probability of attack
        - :math:`A_{att}`: area of attack

    With these we can get the rate of prey consumption:

    .. math::
        W_{prey} = r_{prey} \cdot A_{att} \cdot P_{att}

    And then caloric intake is as simple as:

    .. math::
        C_{in} = W_{prey} / F_{prey}

    Where :math:`F_{prey}` is the number of grams of prey per calorie.

.. admonition:: Core Idea

    Several of these functions are a function of water velocity. This however
    leaves us with a choice - which velocity to use? In this model we assume that
    the water velocity :math:`V_{w}` is the minimum of the maximum swimming velocity of the fish
    and the maximum water velocity in the habitat unit.

.. admonition:: Core Idea

    This model also considers two modes of foraging - movers and stayers. 
    Movers are assumed to have no cover and therefore swim in search of prey.
    Stayers are assumed to have cover and therefore stay in one place and
    strike as prey come by. To capture the relative differences between these
    two modes, we assume that the rate of prey encounter given above is for
    stayers and for movers we have:

    .. math::
        f_{eat} \cdot W_{prey}

.. admonition:: Required Inputs

    This model requires the following derived variables:
        - is the fish a mover or a stayer?
        - :math:`L`: Length of a fish
        - :math:`T`: Temperature (degrees Celsius)
        - :math:`W`: Weight of a fish
        - :math:`V_{wmax}`: Maximum water velocity in the habitat unit

    The following constants
        - :math:`L_{frymin}`: Minimum fry length
        - :math:`F_{prey}`: Grams of prey per calorie

    And the following calibrated parameters:
        - :math:`f_{eat}`: Fraction of prey consumed by movers relative to stayers
        - :math:`L_{att1}`: Length at which :math:`P_{att} = P_{att1}`
        - :math:`L_{att2}`: Length at which :math:`P_{att} = P_{att2}`
        - :math:`P_{att1}`: Probability of attack at :math:`L_{att1}`
        - :math:`P_{att2}`: Probability of attack at :math:`L_{att2}`
        - :math:`m_{drift}`: Drift slope (calories of prey per hour per square centimeter)

.. admonition:: Outputs
         
    It creates the following variables:
        - :math:`V_{max}`: Maximum swimming velocity of the fish
        - :math:`V_{w}`: Water velocity to consider the model (centimeters per second).
        - :math:`P_{att}`: Probability of attack
        - :math:`r_{prey}`: Rate of prey encounter (grams of prey per hour per square centimeter)
        - :math:`A_{att}`: Area of attack (square centimeters)
        - :math:`W_{prey}`: Grams of consumed encountered per hour
        - :math:`C_{in}`: Caloric intake (calories per hour)

.. admonition:: Required Models

    There are some external model dependencies:
        - A model of the probability of capture which is used to determine :math:`A_{att}`.
        - A model of the maximum swimming velocity of the fish which is used to determine :math:`V_{w}`.

.. warning::
    
    - `distance_at_capture_probability` is not implemented yet.
    - `maximum_swimming_velocity` is not implemented yet.

"""

import numpy as np

def distance_at_capture_probability(
    length,
    velocity,
    temperature,
    probability_of_capture
):
    """
    :param length: (:math:`L`)
    :param velocity: (:math:`V_{f}`)
    :param temperature: (:math:`T`)
    :param probability_of_capture: (:math:`P_{cap}`)

    :return: :math:`D_{prey}` maximum distance at which prey can be captured with probability :math:`P_{cap}`
    """
    pass

def maximum_swimming_velocity(weight, temperature):
    """
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)

    :return: :math:`V_{max}`

    Maximum swimming velocity of the fish. This is a function of weight and
    temperature. 
    """
    pass

def water_velocity(maximum_swimming_velocity, maximum_water_velocity):
    """
    :param maximum_swimming_velocity: (:math:`V_{max}`)
    :param maximum_water_velocity: (:math:`V_{wmax}`)

    :return: :math:`V_{w}`

    Water velocity to consider the model. This is the minimum of the maximum
    swimming velocity of the fish and the maximum water velocity in the habitat
    unit.
    """
    return np.minimum(maximum_swimming_velocity, maximum_water_velocity)

def probability_of_attack(
    length, length_fry_min, length_attack_1, length_attack_2,
    prob_attack1, prob_attack2
):
    """
    :param length: (:math:`L`)
    :param length_fry_min: (:math:`L_{frymin}`)
    :param length_attack_1: (:math:`L_{att1}`)
    :param length_attack_2: (:math:`L_{att2}`)
    :param prob_attack1: (:math:`P_{att1}`)
    :param prob_attack2: (:math:`P_{att2}`)

    :return: :math:`P_{att}`

    Probability of attack as a function of fish length. This function is
    based on a logistic curve that is parameterized to pass through
    :math:`P_{att1}` and :math:`P_{att2}` at :math:`L_{att1}` and
    :math:`L_{att2}` respectively. The curve is linear from :math:`L_{frymin}`
    to :math:`L_{att1}`.
    """
    
    # we want this function to be 1 at length < length_fry_min,
    # linear between length_fry_min and length_attack_1,
    # and then a / (L + c) from there on out. however we 
    # want our final curve to pass through prob_attack1 and
    # prob_attack2 at length_attack_1 and length_attack_2.

    c = (prob_attack2 * length_attack_2 - prob_attack1 * length_attack_1) / (prob_attack1 - prob_attack2)
    a = prob_attack1 * (length_attack_1 + c)

    np.piecewise(
        length, 
        [
            length < length_fry_min,
            length_attack_1 > length >= length_fry_min,
            length >= length_attack_1
        ],
        [
            1.0,
            (1 - prob_attack1) * (length_attack_1 - length) / (length_attack_1 - length_fry_min) + prob_attack1,
            (a / (length + c))
        ]
    )

def rate_of_prey_encounter(
    prey_grams, drift_slope, velocity
):
    """
    :param prey_grams: (:math:`F_{prey}`)
    :param drift_slope: (:math:`m_{drift}`)
    :param velocity: (:math:`V_{w}`)

    :return: :math:`r_{prey} = F_{prey} \cdot m_{drift} \cdot V_{w}`

    Note that we're assuming the rate of prey encounter is linearly
    proportional to the water velocity.
    """
    return prey_grams * drift_slope * velocity

def rate_of_prey_consumption(rate_of_prey_encounter, area_of_attack, probability_of_attack):
    """
    :param rate_of_prey_encounter: (:math:`r_{prey}`)
    :param area_of_attack: (:math:`A_{att}`)
    :param probability_of_attack: (:math:`P_{att}`)

    :return: :math:`W_{prey} = r_{prey} \cdot A_{att} \cdot P_{att}`
    
    Number of grams of prey encountered per hour. 
    """
    return rate_of_prey_encounter * area_of_attack * probability_of_attack

def area_of_attack(velocity, length, temperature):
    """
    :param velocity: (:math:`V_{f}`)
    :param length: (:math:`L`)
    :param temperature: (:math:`T`)

    :return: :math:`A_{att}` 

    Find the radius of attack that gives a 90% probability of capture.
    That becomes the radius of the area of attack and all prey within
    are assumed captured if attacked. 
    """
    r = distance_at_capture_probability(length, velocity, temperature, 0.9)
    return np.pi * r ** 2

def caloric_input(rate_of_prey_consumption, prey_grams, mover, mover_fraction):
    """
    :param rate_of_prey_consumption: (:math:`W_{prey}`)
    :param prey_grams: (:math:`F_{prey}`)
    :param mover: boolean
    :param mover_fraction: (:math:`f_{eat}`)

    :return: :math:`C_{in}`

    Caloric intake. Note that for movers we assume they only consume
    a fraction of the prey they encounter.
    """
    if mover:
        return rate_of_prey_consumption * prey_grams * mover_fraction
    else:
        return rate_of_prey_consumption * prey_grams
    
def main(
    mover, weight, length, temperature, velocity,
    prey_grams, minimum_fry_length,
    length_attack_1, length_attack_2,
    prob_attack1, prob_attack2,
    drift_slope, mover_fraction
):
    """
    :param mover: boolean
    :param weight: (:math:`W`)
    :param length: (:math:`L`)
    :param temperature: (:math:`T`)
    :param velocity: (:math:`V_{wmax}`)
    :param prey_grams: (:math:`F_{prey}`)
    :param minimum_fry_length: (:math:`L_{frymin}`)
    :param length_attack_1: (:math:`L_{att1}`)
    :param length_attack_2: (:math:`L_{att2}`)
    :param prob_attack1: (:math:`P_{att1}`)
    :param prob_attack2: (:math:`P_{att2}`)
    :param drift_slope: (:math:`m_{drift}`)
    :param mover_fraction: (:math:`f_{eat}`)

    :return: :math:`C_{in}`
    """
    maximum_swimming_velocity = maximum_swimming_velocity(weight, temperature)
    water_velocity = water_velocity(maximum_swimming_velocity, velocity)
    probability_of_attack = probability_of_attack(length, minimum_fry_length, length_attack_1, length_attack_2, prob_attack1, prob_attack2)
    rate_of_prey_encounter = rate_of_prey_encounter(prey_grams, drift_slope, water_velocity)
    area_of_attack = area_of_attack(water_velocity, length, temperature)
    rate_of_prey_consumption = rate_of_prey_consumption(rate_of_prey_encounter, area_of_attack, probability_of_attack)
    caloric_input = caloric_input(rate_of_prey_consumption, prey_grams, mover, mover_fraction)
    return caloric_input
    