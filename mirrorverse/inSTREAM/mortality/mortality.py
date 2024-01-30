"""
.. admonition:: Core Idea

    There are a few types of mortality considered in this model:
        - high temperature
        - washout (from the stream)
        - stranding
        - reproduction
        - "condition" 
        - angling 
        - stocking

    Overall mortality is then given by:

    .. math::

        M = 1 - \prod {(1 - M_{i})}

.. callgraph:: mirrorverse.inSTREAM.mortality.mortality.main
   :toctree: api
   :zoomable:
   :direction: horizontal

.. warning::
    
    - `condition_mortality_factor` is not implemented yet.
    - `length_mortality_factor` is not implemented yet.
    - `daily_angling_modifier` is not implemented yet.
    - `monthly_angling_modifier` is not implemented yet.
    - `length_angling_modifier` is not implemented yet.
    - `density_angling_modifier` is not implemented yet.
"""

import numpy as np

from mirrorverse.utils import make_cacheable
from mirrorverse.inSTREAM.metabolism.inputs import maximum_swimming_velocity
from mirrorverse.inSTREAM.metabolism.losses import time_spent_foraging


@make_cacheable
def high_temperature_mortality(state):
    """
    :param temperature: (:math:`T`)
    :param low_temp_mortality: (:math:`T_{CTM}`)
    :param high_temp_mortality: (:math:`T_{CTMdel}`)

    :return: :math:`M_{T}`

    0 if temperature is below :math:`T_{CTM}`, 1 if temperature is above :math:`T_{CTMdel}`,
    and a linear interpolation between 0 and 1 if temperature is between :math:`T_{CTM}` and :math:`T_{CTMdel}`.
    """
    temperature = state["temperature"]
    low_temp_mortality = state["low_temp_mortality"]
    high_temp_mortality = state["high_temp_mortality"]
    return np.piecewise(
        temperature,
        [
            temperature < low_temp_mortality,
            low_temp_mortality <= temperature < high_temp_mortality,
            temperature >= high_temp_mortality,
        ],
        [
            0,
            lambda temperature: (temperature - low_temp_mortality)
            / (high_temp_mortality - low_temp_mortality),
            1,
        ],
    )


@make_cacheable
def washout_mortality(state):
    """
    :param bottom_velocity: (:math:`V_{b}`)
    :param cover: has cover

    :return: :math:`M_{W}`

    If the bottom velocity is greater than the fish's maximum swimming velocity and there is no
    cover then the fish gets washed out and dies.
    """
    max_swim_velocity = maximum_swimming_velocity(state)
    return max_swim_velocity < state["bottom_velocity"] and not state["cover"]


@make_cacheable
def stranding_mortality(state):
    """
    :param length: (:math:`L`)
    :param die_depth_fraction: (:math:`d_{die}`)
    :param depth: (:math:`D`)
    :param daily_stranding_mortality_risk: (:math:`P_{strand}`)

    :return: :math:`M_{S}`

    If the fish's length times the die depth fraction is less than the depth of the water then the fish
    is stranded and takes a specific risk of dying.
    """

    if state["length"] * state["die_depth_fraction"] < state["depth"]:
        return state["daily_stranding_mortality_risk"]


@make_cacheable
def reproduction_mortality(state):
    """
    :param reproduced: (:math:`R`)
    :param reproduction_mortality_risk: (:math:`P_{repro}`)

    :return: :math:`M_{R}`

    If the fish has reproduced then it takes on a specific risk of dying.
    """
    if state["reproduced"]:
        return state["reproduction_mortality_risk"]
    return 0


@make_cacheable
def condition_mortality_factor(state):
    """
    :return: :math:`F_{K}`
    """
    return state["weight"] / state["length"] ** 3


@make_cacheable
def length_mortality_factor(state):
    """
    :return: :math:`F_{L}`
    """
    return state["weight"] / state["length"] ** 3


@make_cacheable
def condition_mortality(state):
    """
    :param daily_natural_mortality_rate: (:math:`Z_{24}`)
    :param daily_activity_mortality_rate: (:math:`Z_{act}`)

    :return: :math:`M_{cond}`

    .. math::

        M_{cond} = 1 - e^{-Z}

    .. math::

        Z = F_{K} \cdot F_{L} \cdot (Z_{24} + \\frac{t_{f}}{24} \cdot Z_{act})
    """
    F_K = condition_mortality_factor(state)
    F_L = length_mortality_factor(state)
    t_f = time_spent_foraging(state)
    Z_24 = state["daily_natural_mortality_rate"]
    Z_act = state["daily_activity_mortality_rate"]
    Z = F_K * F_L * (Z_24 + t_f / 24 * Z_act)
    return 1.0 - np.exp(-Z)


@make_cacheable
def daily_angling_modifier(state):
    """
    :return: :math:`f_{day}`

    Higher on weekends than week days
    """
    return state


@make_cacheable
def monthly_angling_modifier(state):
    """
    :return: :math:`f_{month}`

    Highest in June
    """
    return state


@make_cacheable
def length_angling_modifier(state):
    """
    :return: :math:`f_{keep}`

    Higher for larger fish
    """
    return state


@make_cacheable
def density_angling_modifier(state):
    """
    :return: :math:`f_{dens}`

    Higher for higher density
    """
    return state


@make_cacheable
def angling_mortality(state):
    """
    :param species_angling_modifier: (:math:`f_{spp}`)
    :param hooking_mortality_modifier: (:math:`f_{hook}`)
    :param maximum_fishing_mortality_rate: (:math:`P_{fmax}`)

    :return: :math:`M_{A}`

    .. math::

            M_{A} = f_{day} \cdot f_{month} \cdot f_{spp} \cdot (f_{keep} + (1 - f_{keep}) \cdot f_{hook}) \cdot f_{dens} \cdot P_{fmax}
    """
    f_day = daily_angling_modifier(state)
    f_month = monthly_angling_modifier(state)
    f_spp = state["species_angling_modifier"]
    f_keep = length_angling_modifier(state)
    f_hook = state["hooking_mortality_modifier"]
    f_dens = density_angling_modifier(state)
    P_fmax = state["maximum_fishing_mortality_rate"]
    return f_day * f_month * f_spp * (f_keep + (1 - f_keep) * f_hook) * f_dens * P_fmax


@make_cacheable
def stocking_mortality(state):
    """
    :param stocking_mortality_rate: (:math:`P_{HRT}`)

    :return: :math:`M_{HRT}=P_{HRT}`
    """
    return state["stocking_mortality_rate"]


def main(state):
    """
    :return: :math:`M`

    .. math::

        M = 1 - \prod {(1 - M_{i})}
    """
    M_T = high_temperature_mortality(state)
    M_W = washout_mortality(state)
    M_S = stranding_mortality(state)
    M_R = reproduction_mortality(state)
    M_cond = condition_mortality(state)
    M_A = angling_mortality(state)
    M_HRT = stocking_mortality(state)
    return 1 - (1 - M_T) * (1 - M_W) * (1 - M_S) * (1 - M_R) * (1 - M_cond) * (
        1 - M_A
    ) * (1 - M_HRT)
