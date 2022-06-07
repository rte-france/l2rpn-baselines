# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import copy

from l2rpn_baselines.utils import NNParam
from l2rpn_baselines.SACOld.sacOld_NN import SACOld_NN


class SACOld_NNParam(NNParam):
    """
    Do not use this SACOld class that has lots of known (but forgotten) issues.
    
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    .. warning::
        We plan to add SAC based agents relying on external frameworks, such as stable baselines3 or ray / rllib.
        
        We will not code any SAC agent "from scratch".

    Attributes
    ----------
    sizes_value: ``list``
        List of integer, each one representing the size of the hidden layer for the "value" neural network.

    activs_value: ``list``
        List of ``str`` for each hidden layer of the "value" neural network, indicates which hidden layer to use

    sizes_policy: ``list``
        List of integers, each reprenseting the size of the hidden layer for the "policy" network.

    activs_policy: ``list``
        List of ``str``: The activation functions (for each layer) of the policy network

    """
    _int_attr = copy.deepcopy(NNParam._int_attr)
    _float_attr = copy.deepcopy(NNParam._float_attr)
    _str_attr = copy.deepcopy(NNParam._str_attr)
    _list_float = copy.deepcopy(NNParam._list_float)
    _list_str = copy.deepcopy(NNParam._list_str)
    _list_int = copy.deepcopy(NNParam._list_int)

    _list_str += ["activs_value", "activs_policy"]
    _list_int += ["sizes_value", "sizes_policy"]

    nn_class = SACOld_NN

    def __init__(self,
                 action_size,
                 observation_size,  # TODO this might not be usefull
                 sizes,
                 activs,
                 list_attr_obs,
                 sizes_value,
                 activs_value,
                 sizes_policy,
                 activs_policy
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size,  # TODO this might not be usefull
                         sizes,
                         activs,
                         list_attr_obs
                         )
        self.sizes_value = sizes_value
        self.activs_value = activs_value
        self.sizes_policy = sizes_policy
        self.activs_policy = activs_policy
