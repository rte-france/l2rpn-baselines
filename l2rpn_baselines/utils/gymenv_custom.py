# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from abc import abstractmethod
from typing import Tuple, Dict, List
import numpy as np

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.gym_compat import GymEnv


class GymEnvWithHeuristics(GymEnv):
    """This abstract class is used to perform some actions, independantly of a RL
    agent on a grid2op environment.
    
    It can be used, for example, to train an agent (for example a deep-rl agent)
    if you want to use some heuristics at inference time (for example
    you reconnect every powerline that you can.)
    """
    @abstractmethod
    def heuristic_actions(self,
                          g2op_obs: BaseObservation,
                          reward: float,
                          done: bool,
                          info: Dict) -> List[BaseAction]:
        return []
    
    def apply_heuristics_actions(self,
                                 g2op_obs: BaseObservation,
                                 reward: float,
                                 done: bool,
                                 info: Dict ) -> Tuple[BaseObservation, float, bool, Dict]:
        g2op_actions = self.heuristic_actions(g2op_obs, reward, done, info)
        for g2op_act in g2op_actions:
            tmp_obs, tmp_reward, tmp_done, tmp_info = self.init_env.step(g2op_act)
            g2op_obs = tmp_obs
            done = tmp_done
            if tmp_done:
                break
        return g2op_obs, reward, done, info
    
    def step(self, gym_action):
        """[summary]

        Parameters
        ----------
        gym_action : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info
        
    
class GymEnvWithReco(GymEnvWithHeuristics):
    """[summary]

    Parameters
    ----------
    GymEnv : [type]
        [description]
    """
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        return res
