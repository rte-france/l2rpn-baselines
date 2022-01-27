# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from abc import abstractmethod
<<<<<<< HEAD

from grid2op.Agent import BaseAgent
=======
import copy

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
>>>>>>> branch_with_zips


class GymAgent(BaseAgent):
    """
    This class maps a neural network (trained using ray / rllib or stable baselines for example
    
    It can then be used as a "regular" grid2op agent, in a runner, grid2viz, grid2game etc.

    It is also compatible with the "l2rpn baselines" interface.

    Use it only with a trained agent. It does not provide the "save" method and
    is not suitable for training.
<<<<<<< HEAD
    """
    def __init__(self, g2op_action_space, gym_act_space, gym_obs_space, nn_path):
        super().__init__(g2op_action_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space
        self._nn_path = nn_path
        self.nn_model = None
        self.load()

=======
    
    ..info::
        To load a previously saved agent the function `GymAgent.load` will be called
        and you must provide the `nn_path` keyword argument.
        
        To build a new agent, the function `GymAgent.build` is called and
        you must provide the `nn_kwargs` keyword argument.
        
        You cannot set both, you have to set one.
    """
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 *,  # to prevent positional argument
                 nn_path=None,
                 nn_kwargs=None):
        super().__init__(g2op_action_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space
        if nn_path is None and nn_kwargs is None:
            raise RuntimeError("Impossible to build a GymAgent without providing at "
                               "least one of `nn_path` (to load the agent from disk) "
                               "or `nn_kwargs` (to create the underlying agent).")
        if nn_path is not None and nn_kwargs is not None:
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
            
>>>>>>> branch_with_zips
    @abstractmethod
    def get_act(self, gym_obs, reward, done):
        """
        retrieve the action from the NN model
        """
        pass

    @abstractmethod
    def load(self):
        """
<<<<<<< HEAD
        Load the NN models
        """
        pass

    def act(self, observation, reward, done):
=======
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
>>>>>>> branch_with_zips
        gym_obs = self._gym_obs_space.to_gym(observation)
        gym_act = self.get_act(gym_obs, reward, done)
        grid2op_act = self._gym_act_space.from_gym(gym_act)
        return grid2op_act
