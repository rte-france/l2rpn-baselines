# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from grid2op.Agent import BaseAgent


class GymAgent(BaseAgent):
    """
    This class maps a neural network (trained using ray / rllib or stable baselines for example
    
    It can then be used as a "regular" grid2op agent, in a runner, grid2viz, grid2game etc.

    It is also compatible with the "l2rpn baselines" interface.
    """
    def __init__(self, g2op_action_space, gym_act_space, gym_obs_space, nn_path):
        super().__init__(g2op_action_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space
        self._nn_path = nn_path

    