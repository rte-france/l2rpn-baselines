# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


from l2rpn_baselines.utils import GymAgent

from stable_baselines3 import PPO

class SB3Agent(GymAgent):
    def __init__(self, g2op_action_space, gym_act_space, gym_obs_space, nn_path, nn_type=PPO):
        self._nn_type = nn_type
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space, nn_path)
        
    def get_act(self, gym_obs, reward, done):
        action, _ = self.nn_model.predict(gym_obs, deterministic=True)
        return action

    def load(self):
        """
        Load the NN models
        """
        self.nn_model = self._nn_type.load(self._nn_path)
