"""
.. admonition:: Core Idea

    In this model we are concerned with roughly 3 things:
        - :math:`r_{prey}`: rate of prey encounter (grams of prey per hour per square centimeter)
        - :math:`P_{att}`: probability of attack
        - :math:`A_{att}`: area of attack

    With these we can get the rate of prey consumption:

    .. math::
        W_{prey} = r_{prey} \cdot A_{att} \cdot P_{att}

    And then caloric intake (as a rate) is as simple as:

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

.. callgraph:: mirrorverse.inSTREAM.metabolism.inputs.main
   :toctree: api
   :zoomable:
   :direction: vertical

.. warning::
    
    - `distance_at_capture_probability` is not implemented yet.
    - `maximum_swimming_velocity` is not implemented yet.
    - `maximum_consumption_allowable` is not implemented yet.

"""

import numpy as np
from mirrorverse.utils import make_cacheable


@make_cacheable
def maximum_consumption_allowable(state):
    """
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)

    :return: :math:`C_{max}`

    Maximum consumption allowable by the fish. This is a function of weight and
    temperature.
    """
    return state["weight"] * state["temperature"]


@make_cacheable
def distance_at_capture_probability(state):
    """
    :param length: (:math:`L`)
    :param temperature: (:math:`T`)
    :param probability_of_capture: (:math:`P_{cap}`)

    :return: :math:`D_{prey}` maximum distance at which prey can be captured with probability :math:`P_{cap}`
    """
    velocity = water_velocity(state)
    return state["length"]


@make_cacheable
def maximum_swimming_velocity(state):
    """
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)

    :return: :math:`V_{max}`

    Maximum swimming velocity of the fish. This is a function of weight and
    temperature.
    """
    return state["weight"] * state["temperature"]


@make_cacheable
def water_velocity(state):
    """
    :param velocity: (:math:`V_{wmax}`)
    :return: :math:`V_{w}=min(V_{max}, V_{wmax})`

    Water velocity to consider the model. This is the minimum of the maximum
    swimming velocity of the fish and the maximum water velocity in the habitat.
    """
    return np.minimum(maximum_swimming_velocity(state), state["velocity"])


@make_cacheable
def probability_of_attack(state):
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

    length = state["length"]
    length_fry_min = state["minimum_fry_length"]
    length_attack_1 = state["length_attack_1"]
    length_attack_2 = state["length_attack_2"]
    prob_attack1 = state["prob_attack1"]
    prob_attack2 = state["prob_attack2"]

    c = (prob_attack2 * length_attack_2 - prob_attack1 * length_attack_1) / (
        prob_attack1 - prob_attack2
    )
    a = prob_attack1 * (length_attack_1 + c)

    np.piecewise(
        length,
        [
            length < length_fry_min,
            length_attack_1 > length >= length_fry_min,
            length >= length_attack_1,
        ],
        [
            1.0,
            (1 - prob_attack1)
            * (length_attack_1 - length)
            / (length_attack_1 - length_fry_min)
            + prob_attack1,
            (a / (length + c)),
        ],
    )


@make_cacheable
def rate_of_prey_encounter(state):
    """
    :param prey_grams: (:math:`F_{prey}`)
    :param drift_slope: (:math:`m_{drift}`)

    :return: :math:`r_{prey} = F_{prey} \cdot m_{drift} \cdot V_{w}`

    Note that we're assuming the rate of prey encounter is linearly
    proportional to the water velocity.
    """
    return state["prey_grams"] * state["drift_slope"] * water_velocity(state)


@make_cacheable
def area_of_attack(state):
    """
    :param capture_probability: (:math:`P_{cap}`)

    :return: :math:`A_{att}=\pi \cdot D_{prey}^2`

    Find the radius of attack that gives a 90% probability of capture.
    That becomes the radius of the area of attack and all prey within
    are assumed captured if attacked.
    """
    r = distance_at_capture_probability(state)
    return np.pi * r**2


@make_cacheable
def rate_of_prey_consumption(state):
    """
    :return: :math:`W_{prey} = r_{prey} \cdot A_{att} \cdot P_{att}`

    Number of grams of prey encountered per hour.
    """
    return (
        rate_of_prey_encounter(state)
        * area_of_attack(state)
        * probability_of_attack(state)
    )


def main(state):
    """
    :param max_forage_time: (:math:`t_{actmax}`)
    :param prey_grams: (:math:`F_{prey}`)
    :param mover: is the fish a mover or a stayer?
    :param mover_fraction: (:math:`f_{eat}`)

    :return: :math:`C_{in,total} = \min(C_{max}, W_{prey} / F_{prey} \cdot t_{actmax} )`

    Caloric intake. Note that for movers we multiply the rate by :math:`f_{eat}`.
    """
    c_max = maximum_consumption_allowable(state)
    rate_for_day = (
        rate_of_prey_consumption(state)
        / state["prey_grams"]
        * (state["mover_fraction"] if state["mover"] else 1)
    )
    return np.minimum(c_max, rate_for_day * state["max_forage_time"])
