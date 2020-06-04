# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os
import numpy as np

from l2rpn_baselines.utils import NNParam
from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import DuelQLeapNet_NN


class LeapNet_NNParam(NNParam):
    _int_attr = NNParam._int_attr
    _float_attr = NNParam._float_attr
    _str_attr = NNParam._str_attr
    _list_float = NNParam._list_float
    _list_str = NNParam._list_str
    _list_int = NNParam._list_int

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