# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os
from grid2op.Environment import Environment
from l2rpn_baselines.OptimCVXPY.optimCVXPY import OptimCVXPY


def make_agent(env: Environment, dir_path: os.PathLike) -> OptimCVXPY:
    """First example of the function you will need to provide
    to send your agent to l2rpn competitions or
    to use your agent in grid2game.

    Parameters
    ----------
    env : Environment
        _description_
    dir_path : os.PathLike
        _description_

    Returns
    -------
    OptimCVXPY
        _description_
    """
    # TODO read the parameters from a config file !
    agent = OptimCVXPY(env.action_space,
                       env,
                       penalty_redispatching_unsafe=0.,
                       penalty_storage_unsafe=0.1,
                       penalty_curtailment_unsafe=0.01,
                       rho_safe=0.85,
                       rho_danger=0.9,
                       margin_th_limit=0.93,
                       alpha_por_error=0.5,
                       weight_redisp_target=0.,)
    
    return agent
