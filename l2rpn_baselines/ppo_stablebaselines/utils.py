# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


from l2rpn_baselines.utils import GymAgent

try:
    from stable_baselines3 import PPO
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    class PPO(object):
        """
        Do not use, this class is a template when stable baselines3 is not installed.
        
        It represents `from stable_baselines3 import PPO`
        """


class SB3Agent(GymAgent):
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type=PPO,
                 nn_path=None,
                 nn_kwargs=None):
        self._nn_type = nn_type
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space,
                         nn_path=nn_path, nn_kwargs=nn_kwargs)
        
    def get_act(self, gym_obs, reward, done):
        """Retrieve the gym action from the gym observation and the reward. 
        It only (for now) work for non recurrent policy.

        Parameters
        ----------
        gym_obs : gym observation
            The gym observation
        reward : ``float``
            the current reward
        done : ``bool``
            whether the episode is over or not.

        Returns
        -------
        gym action
            The gym action, that is processed in the :func:`GymAgent.act`
            to be used with grid2op
        """
        action, _ = self.nn_model.predict(gym_obs, deterministic=True)
        return action

    def load(self):
        """
        Load the NN model.
        
        In the case of a PPO agent, this is equivalent to perform the:
        
        .. code-block:: python
            
            PPO.load(nn_path)
        """
        self.nn_model = self._nn_type.load(self._nn_path)
        
    def build(self):
        """Create the underlying NN model from scratch.
        
        In the case of a PPO agent, this is equivalent to perform the:
        
        .. code-block:: python
            
            PPO(**nn_kwargs)
        """
        self.nn_model = PPO(**self._nn_kwargs)
