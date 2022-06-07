# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from abc import abstractmethod
import copy
from typing import List, Optional

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

from l2rpn_baselines.utils.gymenv_custom import GymEnvWithHeuristics


class GymAgent(BaseAgent):
    """
    This class maps a neural network (trained using ray / rllib or stable baselines for example
    
    It can then be used as a "regular" grid2op agent, in a runner, grid2viz, grid2game etc.

    It is also compatible with the "l2rpn baselines" interface.

    Use it only with a trained agent. It does not provide the "save" method and
    is not suitable for training.
    
    .. note::
        To load a previously saved agent the function `GymAgent.load` will be called
        and you must provide the `nn_path` keyword argument.
        
        To build a new agent, the function `GymAgent.build` is called and
        you must provide the `nn_kwargs` keyword argument.
    
    Examples
    ---------
    Some examples of such agents are provided in the classes:
    
    - :class:`l2rpn_baselines.PPO_SB3.PPO_SB3` that implements such an agent with the "stable baselines3" RL framework
    - :class:`l2rpn_baselines.PPO_RLLIB.PPO_RLLIB` that implements such an agent with the "ray / rllib" RL framework
    
    Both can benefit from the feature of this class, most notably the possibility to include "heuristics" (such as: 
    "if a powerline can be reconnected, do it" or "do not act if the grid is not in danger")
    
    Notes
    -----
    The main goal of this class is to be able to use "heuristics" (both for training and at inference time) quite simply
    and with out of the box support of external libraries.
    
    All top performers in all l2rpn competitions (as of writing) used some kind of heuristics in their agent (such as: 
    "if a powerline can be reconnected, do it" or "do not act if the grid is not in danger"). This is why we made some 
    effort to develop a generic class that allows to train agents directly using these "heuristics".
    
    This features is split in two parts:
    
    - At training time, the "*heuristics*" are part of the environment. The agent will see only observations that are relevant
      to it (and not the stat handled by the heuristic.)
    - At inference time, the "*heuristics*" of the environment used to train the agent are included in the "agent.act" function.
      If a heuristic has been used at training time, the agent will first "ask" the environment is a heuristic should be
      performed on the grid (in this case it will do it) otherwise it will ask the underlying neural network what to do.
    
    Some examples are provided in the "examples" code (under the "examples/ppo_stable_baselines") repository that 
    demonstrates the use of :class:`l2rpn_baselines.utils.GymEnvWithRecoWithDN` .
    
    """
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 *,  # to prevent positional argument
                 nn_path=None,
                 nn_kwargs=None,
                 gymenv=None,
                 _check_both_set=True,
                 _check_none_set=True):
        super().__init__(g2op_action_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space
        
        self._has_heuristic : bool = False
        self.gymenv : Optional[GymEnvWithHeuristics] = gymenv
        self._action_list : Optional[List] = None
        
        if self.gymenv is not None and isinstance(self.gymenv, GymEnvWithHeuristics):
            self._has_heuristic = True
            self._action_list = []
            
        if _check_none_set and (nn_path is None and nn_kwargs is None):
            raise RuntimeError("Impossible to build a GymAgent without providing at "
                               "least one of `nn_path` (to load the agent from disk) "
                               "or `nn_kwargs` (to create the underlying agent).")
        if _check_both_set and (nn_path is not None and nn_kwargs is not None):
            raise RuntimeError("Impossible to build a GymAgent by providing both "
                               "`nn_path` (*ie* you want load the agent from disk) "
                               "and `nn_kwargs` (*ie* you want to create the underlying agent from these "
                               "parameters).")
        if nn_path is not None:
            self._nn_path = nn_path
        else:
            self._nn_path = None
            
        if nn_kwargs is not None:
            self._nn_kwargs = copy.deepcopy(nn_kwargs)
        else:
            self._nn_kwargs = None
        
        self.nn_model = None
        if nn_path is not None:
            self.load()
        else:
            self.build()
            
    @abstractmethod
    def get_act(self, gym_obs, reward, done):
        """
        retrieve the action from the NN model
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the NN model
        
        ..info:: Only called if the agent has been build with `nn_path` not None and `nn_kwargs=None`
        """
        pass
    
    @abstractmethod
    def build(self):
        """
        Build the NN model.
        
        ..info:: Only called if the agent has been build with `nn_path=None` and `nn_kwargs` not None
        """
        pass
    
    def clean_heuristic_actions(self, observation: BaseObservation, reward: float, done: bool) -> None:
        """This function allows to cure the heuristic actions. 
        
        It is called at each step, just after the heuristic actions are computed (but before they are selected).
        
        It can be used, for example, to reorder the `self._action_list` for example.

        Args:
            observation (BaseObservation): The current observation
            reward (float): the current reward
            done (bool): the current flag "done"
        """
        pass
    
    def act(self, observation: BaseObservation, reward: float, done: bool) -> BaseAction:
        """This function is called to "map" the grid2op world
        into a usable format by a neural networks (for example in a format
        usable by stable baselines or ray/rllib)

        Parameters
        ----------
        observation : BaseObservation
            The grid2op observation
        reward : ``float``
            The reward
        done : function
            the flag "done" by open ai gym.

        Returns
        -------
        BaseAction
            The action taken by the agent, in a form of a grid2op BaseAction.
        
        Notes
        -------
        In case your "real agent" wants to implement some "non learned" heuristic,
        you can also put them here.
        
        In this case the "gym agent" will only be used in particular settings.
        """
        grid2op_act = None
        
        # heuristic part
        if self._has_heuristic:
            if not self._action_list:
                # the list of actions is empty, i querry the heuristic to see if there's something I can do
                self._action_list = self.gymenv.heuristic_actions(observation, reward, done, {})
                
            self.clean_heuristic_actions(observation, reward, done)
            if self._action_list:
                # some heuristic actions have been selected, i select the first one
                grid2op_act = self._action_list.pop(0)
        
        # the heursitic did not select any actions, then ask the NN to do one !
        if grid2op_act is None:
            gym_obs = self._gym_obs_space.to_gym(observation)
            gym_act = self.get_act(gym_obs, reward, done)
            grid2op_act = self._gym_act_space.from_gym(gym_act)
            # fix the action if needed (for example by limiting curtailment and storage)
            grid2op_act = self.gymenv.fix_action(grid2op_act)
            
        return grid2op_act
