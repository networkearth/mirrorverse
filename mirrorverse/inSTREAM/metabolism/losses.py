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


.. admonition:: Core Idea

    That takes care of time. What about the velocities during those times? 

    For movers we assume that they swim at an optimum speed :math:`S_{opt}` and experience a velocity that
    a specified fraction :math:`f_{V}` of the average water velocity :math:`V_{ave}`. They don't spend time
    waiting so we don't need to worry about that.

    For stayers we assume that they experience the flow velocity that is generating their encounter rate :math:`V_{w}`. However
    we have two kinds of stayers. For those with cover we assume :math:`V_{wait}=S_{wait}=0`. For those without cover we assume that
    it is less than the `V_{act}` by a factor `dv_cov`. All other speeds and velocities are considered to be zero.

.. callgraph:: mirrorverse.inSTREAM.metabolism.losses.main
   :toctree: api
   :zoomable:
   :direction: vertical

.. warning::
        
    - `standard_respiration_rate` is not implemented yet.
"""
import numpy as np

from mirrorverse.utils import make_cacheable
from mirrorverse.inSTREAM.metabolism.inputs import (
    maximum_consumption_allowable,
    rate_of_prey_consumption,
    water_velocity,
)


@make_cacheable
def standard_respiration_rate(state):
    """
    :param weight: (:math:`W`)
    :param temperature: (:math:`T`)

    :return: :math:`R_{std}` standard respiration rate
    """
    return state["weight"] * state["temperature"]


@make_cacheable
def time_spent_foraging(state):
    """
    :param mover: is a mover
    :param max_foraging_time: (:math:`t_{actmax}`)
    :param mover_fraction: (:math:`f_{eat}`)
    :param prey_grams: (:math:`F_{prey}`)

    :returns: :math:`t_{act}`

    This returns the maximum time the fish could have been foraging
    given the fact that it can "fill up".
    """
    c_max = maximum_consumption_allowable(state)
    rate_for_day = (
        rate_of_prey_consumption(state)
        / state["prey_grams"]
        * (state["mover_fraction"] if state["mover"] else 1)
    )
    return min(state["max_foraging_time"], c_max / rate_for_day)


@make_cacheable
def time_active(state):
    """
    :param mover: is a mover
    :param mover_fraction: (:math:`f_{eat}`)
    :param prey_gram_per_encounter: (:math:`F_{prey}`)

    :returns: :math:`t_{act}`

    This will be all of the time spent foraging for moving fish
    and 5 seconds for every encounter for waiting fish.
    """
    t_forage = time_spent_foraging(state)
    if state["mover"]:
        return t_forage
    else:
        rate_for_day = rate_of_prey_consumption(state) * (
            state["mover_fraction"] if state["mover"] else 1
        )
        return rate_for_day / state["prey_gram_per_encounter"] * t_forage / 5


@make_cacheable
def time_waiting(state):
    """
    :param mover: is a mover

    :returns: :math:`t_{wait}`

    This will be 0 for moving fish and the time left over after
    foraging for waiting fish.
    """
    if state["mover"]:
        return 0
    return time_spent_foraging(state) - time_active(state)


@make_cacheable
def time_resting(state):
    """
    :param max_foraging_time: (:math:`t_{actmax}`)

    :returns: :math:`t_{rest}`

    This will be the time left over after foraging.
    """
    return state["max_foraging_time"] - time_spent_foraging(state)


@make_cacheable
def velocity_active(state):
    """
    :param mover: is a mover
    :param average_speed: (:math:`V_{ave}`)
    :param velocity_fraction: (:math:`f_{V}`)

    :returns: :math:`V_{act}`

    This will be a fraction of the average speed for moving fish
    and the water velocity for waiting fish.
    """
    if state["mover"]:
        return state["average_speed"] * state["velocity_fraction"]
    else:
        return water_velocity(state)


@make_cacheable
def speed_active(state):
    """
    :param mover: is a mover
    :param optimum_speed: (:math:`S_{opt}`)

    :returns: :math:`S_{act}`

    This will be the optimum speed for moving fish and 0 for waiting fish.
    """
    if state["mover"]:
        return state["optimum_speed"]
    else:
        return 0


@make_cacheable
def velocity_waiting(state):
    """
    :param mover: is a mover
    :param cover: is a stayer with cover
    :param cover_velocity_offset: (:math:`dv_{cov}`)

    :returns: :math:`V_{wait} = V_{act} - dv_{cov}`

    This will only be nonzero for waiting fish with without cover.
    """
    if state["mover"] or state["cover"]:
        return 0
    return velocity_active(state) - state["cover_velocity_offset"]


@make_cacheable
def respiration_activity_fraction(state):
    """
    :param max_foraging_time: (:math:`t_{actmax}`)
    :param respiration_coefficient: (:math:`d_{r}`)

    :returns: :math:`f_{act}`

    .. math::
        f_{act} = t_{act} \cdot e^{d_{r} \cdot (V_{act} + S_{act})} + t_{wait} \cdot e^{d_{r} \cdot (V_{wait} + S_{wait})}
        + t_{rest}{t_{actmax}} \cdot e^{d_{r} \cdot (V_{rest} + S_{rest})} + (24 - t_{actmax})

    """
    V_act = velocity_active(state)
    S_act = speed_active(state)
    t_act = time_active(state)
    V_wait = velocity_waiting(state)
    S_wait = 0
    t_wait = time_waiting(state)
    V_rest = 0
    S_rest = 0
    t_rest = time_resting(state)
    t_max = state["max_foraging_time"]
    dr = state["respiration_coefficient"]
    return (
        t_act * np.exp(dr * (V_act + S_act))
        + t_wait * np.exp(dr * (V_wait + S_wait))
        + t_rest * np.exp(dr * (V_rest + S_rest))
        + (24 - t_max)
    )


@make_cacheable
def other_caloric_losses(state):
    """
    :param f_other: (:math:`f_{other}`)
    :param calories_in: (:math:`C_{in}`)

    :return: :math:`C_{out,other}=f_{other} \cdot C_{in}`
    """
    return state["f_other"] * state["calories_in"]


def main(state):
    """
    :param f_act: (:math:`f_{act}`)

    :returns: :math:`C_{out,total}=C_{out,other}+R_{std} \cdot f_{act}`
    """
    c_out_other = other_caloric_losses(state)
    r_std = standard_respiration_rate(state)
    return c_out_other + r_std * respiration_activity_fraction(state)
