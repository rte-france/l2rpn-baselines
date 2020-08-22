# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float

class SAC_Reward(BaseReward):
    def __init__(self):
        super().__init__()

    def initialize(self, env):
        self.reward_min = dt_float(-10.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env,
                 has_error, is_done,
                 is_illegal, is_ambiguous):
        if has_error:
            return self.reward_min
        if is_illegal or is_ambiguous:
            return (self.reward_min + self.reward_max) / 2.0
        else:
            obs = env.current_obs
            rho = obs.rho[obs.line_status == True]
            rho = np.clip(rho, 0.0, 2.0)
            rho_sq = rho * rho
            inv_rho_sq = np.sum(4.0 - rho_sq)
            r_unit = 0.25 * (inv_rho_sq / dt_float(np.sum(obs.line_status)))
            
            return dt_float(r_unit * self.reward_max)
        
        
