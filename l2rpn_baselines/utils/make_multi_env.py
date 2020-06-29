# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import warnings
from grid2op.Environment import Environment
try:
    from grid2op.Environment import MultiEnvironment
except ImportError:
    # name will be change as of grid2op >= 1.0.0
    try:
        from grid2op.Environment import SingleEnvMultiProcess as MultiEnvironment
    except ImportError as exc:
        raise exc


def make_multi_env(env_init, nb_env):
    """
    This function creates a multi environment compatible with what is expected in the baselines. In particular, it
    adds the observation_space, the action_space and the reward_range attribute.

    The way this function works is explained in the getting_started of grid2op.

    Attributes
    -----------
    env_init: :class:`grid2op.Environment.Environment`
        The environment to duplicates
    nb_env: ``int``
        The number of environment on with which you want to interact at the same time

    Returns
    -------
    res: :class:`grid2op.Environment.MultiEnvironment` or :class:`grid2op.Environment.Environment`
        A copy of the initial environment (if nb_env = 1) or a MultiEnvironment based on the initial environment
        if nb_env >= 2.

    """
    res = None
    nb_env = int(nb_env)

    if nb_env <= 0:
        raise RuntimeError("Impossible to create a negative number of environments")

    if nb_env == 1:
        warnings.warn("You asked to create 1 environment. We didn't use the MultiEnvironment for that. We instead "
                      "created a copy of your initial environment.")
        res = Environment(**env_init.get_kwargs())
    else:
        res = MultiEnvironment(nb_env=nb_env, env=env_init)
        res.observation_space = env_init.observation_space
        res.action_space = env_init.action_space
        res.reward_range = env_init.reward_range
    return res