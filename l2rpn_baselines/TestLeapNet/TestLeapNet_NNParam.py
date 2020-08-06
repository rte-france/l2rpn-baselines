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
from l2rpn_baselines.TestLeapNet.TestLeapNet_NN import TestLeapNet_NN


class TestLeapNet_NNParam(NNParam):
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
    _list_str += ["list_attr_obs_tau", "list_attr_obs_x"]
    _list_float += ["tau_adds", "tau_mults", "x_adds", "x_mults"]
    _list_int += ["tau_dims", "x_dims"]
    nn_class = TestLeapNet_NN

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

                 x_dims,
                 x_adds,
                 x_mults,
                 list_attr_obs_x,

                 input_q_dims,
                 input_q_adds,
                 input_q_mults,
                 list_attr_obs_input_q,

                 list_attr_gm_out,
                 gm_out_dims,
                 gm_out_adds,
                 gm_out_mults,

                 dim_topo,
                 dim_flow,
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size=x_dim + np.sum(tau_dims),
                         sizes=sizes,
                         activs=activs,
                         list_attr_obs=list_attr_obs
                         )
        self.tau_dims = [int(el) for el in tau_dims]

        self.x_dim = x_dim
        self.tau_adds = tau_adds
        self.tau_mults = tau_mults
        self.list_attr_obs_tau = [str(el) for el in list_attr_obs_tau]

        self.x_adds = x_adds
        self.x_mults = x_mults
        self.list_attr_obs_x = [str(el) for el in list_attr_obs_x]
        self.x_dims = [int(el) for el in x_dims]

        # TODO add that for serializing
        self.sizes_enc = [20, 20, 20]
        self.sizes_main = [150, 150, 150]
        self.sizes_out_gm = [100, 40]
        self.sizes_Qnet = [100, 100, 100]

        self.dim_topo = dim_topo
        self.dim_flow = dim_flow

        self.list_attr_obs_input_q = list_attr_obs_input_q
        self.input_q_dims = input_q_dims
        self.input_q_adds = input_q_adds
        self.input_q_mults = input_q_mults

        # grid model output (gm = grid model)
        self.list_attr_gm_out = list_attr_gm_out
        self.gm_out_dims = gm_out_dims
        self.gm_out_adds = gm_out_adds
        self.gm_out_mults = gm_out_mults

    def get_obs_attr(self):
        res = self.list_attr_obs + self.list_attr_obs_input_q
        res += self.list_attr_obs_tau + ["topo_vect"] + self.list_attr_gm_out
        return res