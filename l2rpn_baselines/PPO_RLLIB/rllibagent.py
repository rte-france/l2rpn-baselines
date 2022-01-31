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
    
    Examples
    ---------
    The best way to have such an agent is either to train it:
    
    .. code-block:: python
    
        from l2rpn_baselnes.PPO_RLLIB import train
        agent = train(...)  # see the doc of the `train` function !
        
    Or you can also load it when you evaluate it (after it has been trained !):
    
    .. code-block:: python
    
        from l2rpn_baselnes.PPO_RLLIB import evaluate
        agent = evaluate(...)  # see the doc of the `evaluate` function !
        
    To create such an agent from scratch (NOT RECOMMENDED), you can do:
    
    .. code-block:: python

        import grid2op
        from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace, GymEnv
        from lightsim2grid import LightSimBackend
        
        from l2rpn_baselnes.PPO_RLLIB import RLLIBAgent
            
        env_name = "l2rpn_case14_sandbox"  # or any other name
        
        # customize the observation / action you want to keep
        obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                            "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                            "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                            "storage_power", "storage_charge"]
        act_attr_to_keep = ["redispatch"]
        
        # create the grid2op environment
        env = grid2op.make(env_name, backend=LightSimBackend())
        
        # define the action space and observation space that your agent
        # will be able to use
        gym_observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
        gym_action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_attr_to_keep)

        # define the configuration for the environment
        env_config = {"env_name": env.env_name,
                      "backend_class": LightSimBackend,
                      "obs_attr_to_keep": obs_attr_to_keep,
                      "act_attr_to_keep": act_attr_to_keep, 
                      # other type of parameters used in the "grid2op.make"
                      # function eg:
                      # "param": ...
                      # "reward_class": ...
                      # "other_reward": ...
                      # "difficulty": ...
                      }

        # now define the configuration for the PPOTrainer
        env_config_ppo = {
            # config to pass to env class
            "env_config": env_config,
            #neural network config
            "lr": 1e-4, # learning_rate
            "model": {
                "fcnet_hiddens": [100, 100, 100],  # neural net architecture
            },
            # other keyword arguments
        }
        
        # create a grid2gop agent based on that (this will reload the save weights)
        grid2op_agent = RLLIBAgent(env.action_space,
                                    gym_action_space,
                                    gym_observation_space,
                                    nn_config=env_config_ppo,
                                    nn_path=None  # don't load it from anywhere
                                    )
        
        # use it
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        grid2op_act = grid2op_agent.act(obs, reward, done)
        obs, reward, done, info = env.step(grid2op_act)
        
        # NB: the agent above is NOT trained ! So it's likely to output "random" 
        # actions !
                                   
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

if __name__ == "__main__":
    import grid2op
    from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace
    from lightsim2grid import LightSimBackend
    
    env_name = "l2rpn_case14_sandbox"  # or any other name
    obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                        "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                        "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                        "storage_power", "storage_charge"]
    act_attr_to_keep = ["redispatch"]
    
    # create the grid2op environment
    env = grid2op.make(env_name, backend=LightSimBackend())
    
    # define the action space and observation space that your agent
    # will be able to use
    gym_observation_space =  BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
    gym_action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_attr_to_keep)

    # define the configuration for the environment
    env_config = {"env_name": env.env_name,
                  "backend_class": LightSimBackend,
                  "obs_attr_to_keep": obs_attr_to_keep,
                  "act_attr_to_keep": act_attr_to_keep, 
                    # other type of parameters used in the "grid2op.make"
                    # function eg:
                    # "param": ...
                    # "reward_class": ...
                    # "other_reward": ...
                    # "difficulty": ...
                    }

    # now define the configuration for the PPOTrainer
    env_config_ppo = {
        # config to pass to env class
        "env_config": env_config,
        #neural network config
        "lr": 1e-4, # learning_rate
        "model": {
            "fcnet_hiddens": [100, 100, 100],  # neural net architecture
        },
        # other keyword arguments
    }
    
    # create a grid2gop agent based on that (this will reload the save weights)
    grid2op_agent = RLLIBAgent(env.action_space,
                                gym_action_space,
                                gym_observation_space,
                                nn_config=env_config_ppo,
                                nn_path=None  # don't load it from anywhere
                                )
    
    # use it
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    grid2op_act = grid2op_agent.act(obs, reward, done)
    obs, reward, done, info = env.step(grid2op_act)
    