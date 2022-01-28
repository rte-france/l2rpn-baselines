# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import warnings
import os
import json
from typing import List, Optional

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
        
        
default_obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                            "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                            "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                            "storage_power", "storage_charge"]


default_act_attr_to_keep = ["redispatch", "curtail", "set_storage"]


def remove_non_usable_attr(grid2openv, act_attr_to_keep: List[str]) -> List[str]:
    """This function modifies the attribute (of the actions)
    to remove the one that are non usable with your gym environment.
    
    If only filters things if the default variables are used 
    (see _default_act_attr_to_keep)

    Parameters
    ----------
    grid2openv : grid2op.Environment.Environment
        The used grid2op environment
    act_attr_to_keep : List[str]
        The attributes of the actions to keep.

    Returns
    -------
    List[str]
        The same as `act_attr_to_keep` if the user modified the default.
        Or the attributes usable by the environment from the default list.
        
    """
    modif_attr = act_attr_to_keep
    if act_attr_to_keep == default_act_attr_to_keep:
        # by default, i remove all the attributes that are not supported by the action type
        # i do not do that if the user specified specific attributes to keep. This is his responsibility in
        # in this case
        modif_attr = []
        for el in act_attr_to_keep:
            if grid2openv.action_space.supports_type(el):
                modif_attr.append(el)
            else:
                warnings.warn(f"attribute {el} cannot be processed by the allowed "
                                "action type. It has been removed from the "
                                "gym space as well.")
    return modif_attr


def save_used_attribute(save_path: Optional[str],
                        name: str,
                        obs_attr_to_keep: List[str],
                        act_attr_to_keep: List[str]) -> bool:
    """Serialize, as jon the obs_attr_to_keep and act_attr_to_keep
    
    This is typically called in the `train` function.

    Parameters
    ----------
    save_path : Optional[str]
        where to save the used attributes (put ``None`` if you don't want to
        save it)
    name : str
        Name of the model
    obs_attr_to_keep : List[str]
        List of observation attributes to keep
    act_attr_to_keep : List[str]
        List of action attributes to keep

    Returns
    -------
    bool
        whether the data have been saved or not
    """
    res = False
    if save_path is not None:
        my_path = os.path.join(save_path, name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(my_path):
            os.mkdir(my_path)

        with open(os.path.join(my_path, "obs_attr_to_keep.json"), encoding="utf-8", mode="w") as f:
            json.dump(fp=f, obj=obs_attr_to_keep)
        with open(os.path.join(my_path, "act_attr_to_keep.json"), encoding="utf-8", mode="w") as f:
            json.dump(fp=f, obj=act_attr_to_keep)
        res = True
    return res
      
            
class SB3Agent(GymAgent):
    """This class represents the Agent (directly usable with grid2op framework)

    This agents uses the stable-baselines3 `nn_type` (by default PPO) as
    the neural network to take decisions on the grid.
    
    To be built, it requires:
    
    - `g2op_action_space`: a grid2op action space (used for initializing the grid2op agent)
    - `gym_act_space`: a gym observation space (used for the neural networks)
    - `gym_obs_space`: a gym action space (used for the neural networks)

    It can also accept different types of parameters:
    
    - `nn_type`: the type of "neural network" from stable baselines (by default PPO)
    - `nn_path`: the path where the neural network can be loaded from
    - `nn_kwargs`: the parameters used to build the neural network from scratch.
    
    Exactly one of `nn_path` and `nn_kwargs` should be provided. No more, no less.
    """
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
