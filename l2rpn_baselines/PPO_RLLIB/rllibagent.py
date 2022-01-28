# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import copy
import os
import re
from typing import List, Optional

from l2rpn_baselines.utils import GymAgent
from l2rpn_baselines.PPO_RLLIB.env_rllib import Env_RLLIB

try:
    from ray.rllib.agents.ppo import PPOTrainer
    _CAN_USE_RLLIB = True
except ImportError:
    _CAN_USE_RLLIB = False
    
    class PPOTrainer(object):
        """
        Do not use, this class is a template when rllib is not installed.
        
        It represents `from ray.rllib.agents.ppo import PPOTrainer`
        """
        

class RLLIBAgent(GymAgent):
    """This class represents the Agent (directly usable with grid2op framework)

    This agents uses the stable baseline `nn_type` (by default PPOTrainer) as
    the neural network to take decisions on the grid.
    
    To be built, it requires:
    
    - `g2op_action_space`: a grid2op action space (used for initializing the grid2op agent)
    - `gym_act_space`: a gym observation space (used for the neural networks)
    - `gym_obs_space`: a gym action space (used for the neural networks)
    - `nn_config`: the parameters used to build the rllib "trainer" (the thing
      tagged "nn_config" in rllib)
    
    It can also accept different types of parameters:
    
    - `nn_type`: the type of "neural network" from rllib (by default PPOTrainer)
    - `nn_path`: the path where the neural network can be loaded from
    
    For this class `nn_config` is mandatory. The trainer is built with:
    
    .. code-block:: python
    
        from l2rpn_baselines.PPO_RLLIB import Env_RLLIB
        PPOTrainer(env=Env_RLLIB, config=nn_config)
    
    """
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_config,
                 nn_type=PPOTrainer,
                 nn_path=None,
                 ):
        if not _CAN_USE_RLLIB:
            raise ImportError("Cannot import ray[rllib]. Impossible to use this class.")
        
        self._nn_type = nn_type
        if nn_config is None:
            raise RuntimeError("For RLLIB agent you need to provide nn_kwargs")
        self._nn_config = nn_config
        
        nn_kwargs = {"env": Env_RLLIB,
                     "config": nn_config
                    }
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space,
                         nn_path=nn_path, nn_kwargs=nn_kwargs,
                         _check_both_set=False,)
        
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
        action = self.nn_model.compute_single_action(gym_obs)
        return action

    def load(self):
        """
        Load the NN model.
        
        In the case of a PPO agent, this is equivalent to perform the:
        
        .. code-block:: python
            
            PPOTrainer.restore(nn_path)
            
        """
        self.build()
        chkts = sorted(os.listdir(self._nn_path))
        last_chkts = [re.match("checkpoint_[0-9]+$", el) is not None
                     for el in chkts]
        last_chkts = [el for el, ok_ in zip(chkts, last_chkts) if ok_]
        last_chkt_path = last_chkts[-1]
        last_chkt_path = os.path.join(self._nn_path, last_chkt_path)
        possible_chkt = [el for el in os.listdir(last_chkt_path)
                         if re.match(".*.tune_metadata$", el) is not None]
        assert len(possible_chkt)
        last_chkt = possible_chkt[-1]
        last_chkt = re.sub(r"\.tune_metadata$", "", last_chkt)
        self.nn_model.restore(checkpoint_path=os.path.join(last_chkt_path, last_chkt))
        
    def build(self):
        """Create the underlying NN model from scratch.
        
        In the case of a PPO agent, this is equivalent to perform the:
        
        .. code-block:: python
            
            PPOTrainer(env= Env_RLLIB, config=nn_config)
            
        """
        self.nn_model = PPOTrainer(**self._nn_kwargs)
