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


def make_agent(env: Environment,
               dir_path: os.PathLike,
               lines_x_pu=None,
               margin_th_limit: float=0.9,
               alpha_por_error: float=0.5,
               rho_danger: float=0.95,
               rho_safe: float=0.85,
               penalty_curtailment_unsafe: float=0.1,
               penalty_redispatching_unsafe: float=0.03,
               penalty_storage_unsafe: float=0.3,
               penalty_curtailment_safe: float=0.0,
               penalty_redispatching_safe: float=0.0,
               weight_redisp_target: float=1.0,
               weight_storage_target: float=1.0,
               weight_curtail_target: float=1.0,
               penalty_storage_safe: float=0.0,
               margin_rounding: float=0.01,
               margin_sparse: float=5e-3,
               ) -> OptimCVXPY:
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
                       lines_x_pu=lines_x_pu,
                       margin_th_limit=margin_th_limit,
                       alpha_por_error=alpha_por_error,
                       rho_danger=rho_danger,
                       rho_safe=rho_safe,
                       penalty_curtailment_unsafe=penalty_curtailment_unsafe,
                       penalty_redispatching_unsafe=penalty_redispatching_unsafe,
                       penalty_storage_unsafe=penalty_storage_unsafe,
                       penalty_curtailment_safe=penalty_curtailment_safe,
                       penalty_redispatching_safe=penalty_redispatching_safe,
                       weight_redisp_target=weight_redisp_target,
                       weight_storage_target=weight_storage_target,
                       weight_curtail_target=weight_curtail_target,
                       penalty_storage_safe=penalty_storage_safe,
                       margin_rounding=margin_rounding,
                       margin_sparse=margin_sparse)
    
    return agent
