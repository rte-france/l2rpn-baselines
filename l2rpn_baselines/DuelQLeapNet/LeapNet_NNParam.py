# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os
import numpy as np
import copy

from l2rpn_baselines.utils import NNParam
from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import DuelQLeapNet_NN


class LeapNet_NNParam(NNParam):
    """
    This class implements the type of parameters used by the DuelQLeapNet model.

    More information on the leap net can be found at `Leap Net on Github <https://github.com/BDonnot/leap_net>`_

    Attributes
    -----------
    x_dim: ``int``
        Dimension of the input `x`

    list_attr_obs_tau: ``str``
        List of the name of the observation variable that will be used as vector tau to performs the leaps.

    tau_dims: ``list``
        List of ``int``. For each variable considered as a `tau` specify its dimension here.

    tau_adds: ``list``
        List of ``float`` if you want to add something to the value of the observation you receive. For example if you
        know the observation you will receive is either 1 or 2 but prefer these number to be 0 and 1, you can set
        the relevant `tau_adds` to "-1"

    tau_mults: ``list``
        List of ``float``. Same as above if for multiplicative term. If you want to multiply the number you get by a
        specific number (for example if you have numbers in the range 0,10 but would rather have numbers in the range
        0,1, you can set the `tau_mults` number to `0.1`

    """
    _int_attr = copy.deepcopy(NNParam._int_attr)
    _float_attr = copy.deepcopy(NNParam._float_attr)
    _str_attr = copy.deepcopy(NNParam._str_attr)
    _list_float = copy.deepcopy(NNParam._list_float)
    _list_str = copy.deepcopy(NNParam._list_str)
    _list_int = copy.deepcopy(NNParam._list_int)

    _int_attr += ["x_dim"]
    _list_str += ["list_attr_obs_tau"]
    _list_float += ["tau_adds", "tau_mults"]
    _list_int += ["tau_dims"]
    nn_class = DuelQLeapNet_NN

    def __init__(self,
                 action_size,
                 observation_size,  # not used here for retro compatibility with NNParam.from_dict
                 sizes,
                 activs,
                 x_dim,
                 list_attr_obs,
                 tau_dims,
                 tau_adds,
                 tau_mults,
                 list_attr_obs_tau,
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size=x_dim + np.sum(tau_dims),  # TODO this might not be usefull
                         sizes=sizes,
                         activs=activs,
                         list_attr_obs=list_attr_obs
                         )
        self.tau_dims = [int(el) for el in tau_dims]
        self.list_attr_obs_tau = [str(el) for el in list_attr_obs_tau]
        self.x_dim = x_dim
        self.tau_adds = tau_adds
        self.tau_mults = tau_mults

    def get_obs_attr(self):
        return self.list_attr_obs + self.list_attr_obs_tau