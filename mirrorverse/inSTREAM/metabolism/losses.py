"""
.. admonition:: Core Idea

    The authors of this model acknowledge several differents kinds of metabolic losses:
        - Egestion
        - Excretion
        - Specific Dynamic Action
        - Respiration

    However they note that studies have shown the first three can be grouped together and
    represented as a fraction of the energy consumed. 

    This therefore leaves us with only respiration to consider in any real depth. For this
    they break things into two pieces. :math:`R_{std}` or the standard respiration rate and
    then a time weighted sum of factors that determine the increase in respriration rate due
    to activity. 

    .. math::
        R_{total} = R_{std} \cdot f_{act}

.. admonition:: Core Idea

    Understanding the activity factor requires a couple of things. First we need to understand
    speeds and velocities that the fish is experiencing during its activities. Then we need
    to understand the time taken in each activity. 

    For this there are three designations:

    - movers
    - stayers with cover
    - stayers without cover

    We'll begin with time. Time is divided into three categories - :math:`t_{act}`, :math:`t_{rest}`, and :math:`t_{wait}`.
    It is assumed that all fish spend as much time hunting as they possibly can. This means that they will spend up to 
    :math:`t_{actmax}` hunting. Unless they "fill up" before then. In this case whatever time is left with be spent in the
    :math:`t_{rest}` category. 

    As for :math:`t_{act}` and :math:`t_{wait}` it is assumed that for movers `t_{wait}=0`. For stayers it is whatever time
    is leftover after :math:`t_{act}`. So how is :math:`t_{act}` determined? Well we simply assume that 5s will be spent on
    every prey encounter. 

.. callgraph:: mirrorverse.inSTREAM.metabolism.losses.main
   :toctree: api
   :zoomable:
   :direction: vertical

.. warning::
        
    - `standard_respiration_rate` is not implemented yet.
"""

from mirrorverse.inSTREAM.metabolism.inputs import (
    maximum_consumption_allowable,
    rate_of_prey_consumption,
)


def standard_respiration_rate(weight, temperature):
    """
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)

    :return: :math:`R_{std}` standard respiration rate
    """
    return weight * temperature


def time_spent_foraging(
    mover,
    max_foraging_time,
    prey_grams,
    drift_slope,
    weight,
    temperature,
    velocity,
    length,
    minimum_fry_length,
    length_attack_1,
    length_attack_2,
    prob_attack1,
    prob_attack2,
    mover_fraction,
):
    """
    :param mover: is a mover
    :param max_foraging_time: (:math:`t_{actmax}`)

    :returns: :math:`t_{act}`

    This returns the maximum time the fish could have been foraging
    given the fact that it can "fill up".

    Pass Through:

    :param prey_grams: (:math:`W_{prey}`)
    :param drift_slope: (:math:`S_{drift}`)
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)
    :param velocity: (:math:`V_{f}`)
    :param length: (:math:`L`)
    :param minimum_fry_length: (:math:`L_{min}`)
    :param length_attack_1: (:math:`L_{att1}`)
    :param length_attack_2: (:math:`L_{att2}`)
    :param prob_attack1: (:math:`P_{att1}`)
    :param prob_attack2: (:math:`P_{att2}`)
    :param mover_fraction: (:math:`f_{eat}`)
    """
    c_max = maximum_consumption_allowable(weight, temperature)
    rate_for_day = (
        rate_of_prey_consumption(
            prey_grams,
            drift_slope,
            weight,
            temperature,
            velocity,
            length,
            minimum_fry_length,
            length_attack_1,
            length_attack_2,
            prob_attack1,
            prob_attack2,
        )
        / prey_grams
        * (mover_fraction if mover else 1)
    )
    return min(max_foraging_time, c_max / rate_for_day)


def time_active():
    t_forage = time_spent_foraging(
        mover,
        max_foraging_time,
        prey_grams,
        drift_slope,
        weight,
        temperature,
        velocity,
        length,
        minimum_fry_length,
        length_attack_1,
        length_attack_2,
        prob_attack1,
        prob_attack2,
        mover_fraction,
    )
    if mover:
        return t_forage
    else:
        rate_for_day = (
            rate_of_prey_consumption(
                prey_grams,
                drift_slope,
                weight,
                temperature,
                velocity,
                length,
                minimum_fry_length,
                length_attack_1,
                length_attack_2,
                prob_attack1,
                prob_attack2,
            )
            / prey_grams
            * (mover_fraction if mover else 1)
        )
        return rate_for_day * prey_per_gram * t_forage / 5


def other_caloric_losses(f_other, calories_in):
    """
    :param f_other: (:math:`f_{other}`)
    :param calories_in: (:math:`C_{in}`)

    :return: :math:`C_{out,other}=f_{other} \cdot C_{in}`
    """
    return f_other * calories_in


def main(f_act, f_other, calories_in, weight, temperature):
    """
    :param f_act: (:math:`f_{act}`)

    :returns: :math:`C_{out,total}=C_{out,other}+R_{std} \cdot f_{act}`

    Pass Through:

    :param f_other: (:math:`f_{other}`)
    :param calories_in: (:math:`C_{in}`)
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)
    """
    c_out_other = other_caloric_losses(f_other, calories_in)
    r_std = standard_respiration_rate(weight, temperature)
    return r_std * f_act
