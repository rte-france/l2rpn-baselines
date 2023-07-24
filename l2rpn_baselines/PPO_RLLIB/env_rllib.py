# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


import grid2op
from grid2op.gym_compat import (BoxGymActSpace,
                                BoxGymObsSpace,
                                GymEnv
                                )
from grid2op.gym_compat.utils import ALL_ATTR_CONT
from grid2op.gym_compat.box_gym_obsspace import ALL_ATTR_OBS
from l2rpn_baselines.PPO_SB3 import remove_non_usable_attr


if GymEnv._gymnasium:
    import gymnasium as gym
    from gymnasium.spaces import Box
else:
    import gym
    from gym.spaces import Box


class Env_RLLIB(gym.Env):
    """
    This class represents the Environment usable
    from rllib, mapping a grid2op environment.
    
    It is primarily made to serve as example of what is possible to achieve.
    You might probably want to customize this environment to your specific
    needs.

    This agents uses the rllib framework to code for 
    a neural network.
    
    .. warning::
        A grid2op environment is created  when this agent is made. We found
        out rllib worked better this way.
    
    .. alert::
        This class inherits either from `gymnasium.Env` or 
        `gym.Env` depending on grid2op underlying `GymEnv` base class.
        
        Please consult grid2op docs for more information.
        
        By default it should be a gymnasium environment !
    
    To be built, it requires the `env_config` parameters. This parameter is a 
    dictionnary with keys:
    
    - "env_name": the name of the environment you want to make
    - "obs_attr_to_keep": the attributes of the observation you want to use
      in the gym observation space (gym observation space is converted
      to a Box)
    - "act_attr_to_keep" : the attributes of the action you want to use in 
      the gym action space (gym action space is also converted to a 
      Box)
    - "backend_class": the type of backed to use
    - "backend_kwargs": the extra key word arguments to used when creating 
      the backend
    - all other arguments are passed to `grid2op.make(...)` function
    
    """
    
    def __init__(self, env_config):
        # boilerplate code...
        # retrieve the information
        if not "env_name" in env_config:
            raise RuntimeError("The configuration for RLLIB should provide the env name")
        
        nm_env = env_config["env_name"]
        del env_config["env_name"]
        obs_attr_to_keep = None
        if "obs_attr_to_keep" in env_config:
            obs_attr_to_keep = env_config["obs_attr_to_keep"]
            del  env_config["obs_attr_to_keep"]
        act_attr_to_keep = None
        if "act_attr_to_keep" in env_config:  
            act_attr_to_keep = env_config["act_attr_to_keep"]
            del  env_config["act_attr_to_keep"]
            
        if "backend_class" in env_config:
            backend_kwargs = {}
            if "backend_kwargs" in env_config:
                backend_kwargs = env_config["backend_kwargs"]
                del env_config["backend_kwargs"]
            backend = env_config["backend_class"](**backend_kwargs)
            del  env_config["backend_class"]
        else:
            try:
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
            except ImportError as exc_:
                from grid2op.Backend import PandaPowerBackend
                backend = PandaPowerBackend()
                
            
        # 1. create the grid2op environment
        self.env_glop = grid2op.make(nm_env, backend=backend, **env_config)
        # clean the attribute
        act_attr_to_keep = remove_non_usable_attr(self.env_glop, act_attr_to_keep)
        
        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)

        # 3. customize action space and observation space
        if obs_attr_to_keep is None:
            obs_attr_to_keep = ALL_ATTR_OBS
        
        self.env_gym.observation_space.close()
        self.env_gym.observation_space = BoxGymObsSpace(self.env_glop.observation_space,
                                                        attr_to_keep=obs_attr_to_keep)
        
        if act_attr_to_keep is None:    
            act_attr_to_keep = ALL_ATTR_CONT
        self.env_gym.action_space.close()
        self.env_gym.action_space = BoxGymActSpace(self.env_glop.action_space,
                                                    attr_to_keep=act_attr_to_keep)

        # 4. specific to rllib
        self.action_space = Box(low=self.env_gym.action_space.low,
                                high=self.env_gym.action_space.high,
                                shape=self.env_gym.action_space.shape)
        self.observation_space = Box(low=self.env_gym.observation_space.low,
                                     high=self.env_gym.observation_space.high,
                                     shape=self.env_gym.observation_space.shape)

    def reset(self, *, seed=None, options=None):
        tmp = self.env_gym.reset(seed=seed, options=options)
        return tmp
    
    def step(self, action):
        return self.env_gym.step(action)
