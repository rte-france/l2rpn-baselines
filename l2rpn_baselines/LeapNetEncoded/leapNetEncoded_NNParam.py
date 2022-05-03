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
from l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NN import LeapNetEncoded_NN


class LeapNetEncoded_NNParam(NNParam):
    """
    This class implements the type of parameters used by the :class:`LeapNetEncoded` model.

    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    More information on the leap net can be found at `Leap Net on Github <https://github.com/BDonnot/leap_net>`_

    Attributes
    -----------
    list_attr_obs:
        currently ot used
    sizes:
        currently not used
    activs:
        currently not used
    x_dim:
        currently not used

    list_attr_obs_x:
        list of the attribute of the observation that serve as input of the grid model
        (we recommend ["prod_p", "prod_v", "load_p", "load_q"])
    list_attr_obs_gm_out:
        list of the attribute of the observation that serve as output for the grid model
        (we recommend ["a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v"] + li_attr_obs_X)
        though "rho" can be equally good an improve computation time
    list_attr_obs_input_q:
        list of the attribute of the observation that serve as input (other that the embedding of the
        grid state) for the Q network (we recommend to have here anything "time related" for example
        ["time_before_cooldown_line", "time_before_cooldown_sub",  "actual_dispatch",
        "target_dispatch",  "day_of_week", "hour_of_day",  "minute_of_hour"] etc.
    list_attr_obs_tau:
        If you chose to encode your q network as a leap net it self, then you can put here the attribute
        you would like the leap net to act on ( ["line_status", "timestep_overflow"] for example)
    dim_topo: ``int``
        Dimension of the topology vector (init it with `env.dim_topo`)

    Examples
    --------
    All other attributes need to be created once by a call to :func:`l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NNParam.LeapNetEncoded_NNParam.compute_dims`:

    .. code-block:: python

        nn_archi.compute_dims(env)
        nn_archi.center_reduce(env)

    These calls will set up all the attribute that are not set, and register this model to use
    input data approximately in [-1,1] interval.


    """
    _int_attr = copy.deepcopy(NNParam._int_attr)
    _float_attr = copy.deepcopy(NNParam._float_attr)
    _str_attr = copy.deepcopy(NNParam._str_attr)
    _list_float = copy.deepcopy(NNParam._list_float)
    _list_str = copy.deepcopy(NNParam._list_str)
    _list_int = copy.deepcopy(NNParam._list_int)

    _int_attr += ["x_dim", "dim_topo"]
    _list_str += ["list_attr_obs_tau", "list_attr_obs_x", "list_attr_obs_input_q",
                  "list_attr_obs_gm_out"]
    _list_float += ["tau_adds", "tau_mults", "x_adds", "x_mults",
                    "input_q_adds", "input_q_mults",
                    "gm_out_adds", "gm_out_mults"]
    _list_int += ["tau_dims", "x_dims", "gm_out_dims", "input_q_dims",
                  "sizes_enc", "sizes_main", "sizes_out_gm", "sizes_Qnet"]
    nn_class = LeapNetEncoded_NN

    def __init__(self,
                 action_size,
                 observation_size,  # not used here for retro compatibility with NNParam.from_dict
                 sizes,
                 activs,
                 x_dim,

                 list_attr_obs,
                 list_attr_obs_tau,
                 list_attr_obs_x,
                 list_attr_obs_input_q,
                 list_attr_obs_gm_out,

                 dim_topo,

                 sizes_enc=(20, 20, 20),
                 sizes_main=(150, 150, 150),
                 sizes_out_gm=(100, 40),
                 sizes_Qnet=(100, 100, 100),

                 input_q_adds=None,
                 input_q_mults=None,
                 gm_out_adds=None,
                 gm_out_mults=None,
                 tau_adds=None,
                 tau_mults=None,
                 x_adds=None,
                 x_mults=None,

                 tau_dims=None,
                 x_dims=None,
                 gm_out_dims=None,
                 input_q_dims=None,
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size=0,  # not used
                         sizes=sizes,
                         activs=activs,
                         list_attr_obs=list_attr_obs
                         )

        self.x_dim = x_dim

        self.list_attr_obs_tau = [str(el) for el in list_attr_obs_tau]
        self._define_adds_mults(tau_adds, "tau_adds", list_attr_obs_tau, 0.)
        self._define_adds_mults(tau_mults, "tau_mults", list_attr_obs_tau, 1.)

        self.list_attr_obs_x = [str(el) for el in list_attr_obs_x]
        self._define_adds_mults(x_adds, "x_adds", list_attr_obs_x, 0.)
        self._define_adds_mults(x_mults, "x_mults", list_attr_obs_x, 1.)

        self.list_attr_obs_input_q = [str(el) for el in list_attr_obs_input_q]
        self._define_adds_mults(input_q_adds, "input_q_adds", list_attr_obs_input_q, 0.)
        self._define_adds_mults(input_q_mults, "input_q_mults", list_attr_obs_input_q, 1.)

        self.list_attr_obs_gm_out = [str(el) for el in list_attr_obs_gm_out]
        self._define_adds_mults(gm_out_adds, "gm_out_adds", list_attr_obs_gm_out, 0.)
        self._define_adds_mults(gm_out_mults, "gm_out_mults", list_attr_obs_gm_out, 1.)

        # sizes of the neural network "blccks"
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out_gm = sizes_out_gm
        self.sizes_Qnet = sizes_Qnet

        # dimension of the topogly and number of powerline
        self.dim_topo = dim_topo

        # dimension of the space (can be computed in the self.compute_dims)
        self.input_q_dims = input_q_dims
        self.gm_out_dims = gm_out_dims
        self.x_dims = x_dims
        self.tau_dims = tau_dims

    def get_obs_attr(self):
        """
        Retrieve the list of the observation attributes that are used for this model.
        """
        res = self.list_attr_obs_x + self.list_attr_obs_input_q
        res += self.list_attr_obs_tau + ["topo_vect"] + self.list_attr_obs_gm_out
        return res

    def compute_dims(self, env):
        """Compute the dimension of the observations (dimension of x and tau)

        Parameters
        ----------
        env : a grid2op environment
            A grid2op environment
        """
        self.tau_dims = [int(LeapNetEncoded_NNParam.get_obs_size(env, [el])) for el in self.list_attr_obs_tau]
        self.x_dims = [int(LeapNetEncoded_NNParam.get_obs_size(env, [el])) for el in self.list_attr_obs_x]
        self.gm_out_dims = [int(LeapNetEncoded_NNParam.get_obs_size(env, [el])) for el in self.list_attr_obs_gm_out]
        self.input_q_dims = [int(LeapNetEncoded_NNParam.get_obs_size(env, [el])) for el in self.list_attr_obs_input_q]

    def _define_adds_mults(self, vector, varname, attr_composed, default_val):
        if vector is None:
            vector = [float(default_val) for _ in attr_composed]
        setattr(self, varname, vector)

    def center_reduce(self, env):
        """
        Compute some basic statistics for x and tau
        """
        self._center_reduce_vect(env.get_obs(), "x")
        self._center_reduce_vect(env.get_obs(), "tau")
        self._center_reduce_vect(env.get_obs(), "gm_out")
        self._center_reduce_vect(env.get_obs(), "input_q")

    def _get_adds_mults_from_name(self, obs, attr_nm):
        add_tmp, mult_tmp = super()._get_adds_mults_from_name(obs, attr_nm)
        if attr_nm in ["line_status"]:
            # transform time step overflow into (1. - timestep_overflow) [similar to the leap net papers]
            # 0 powerline is connected, 1 powerline is NOT connected
            add_tmp = -1.0
            mult_tmp = -1.0
        return add_tmp, mult_tmp
