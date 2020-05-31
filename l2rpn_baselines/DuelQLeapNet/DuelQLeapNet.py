# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
from l2rpn_baselines.utils import DeepQAgent
from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import DuelQLeapNet_NN

DEFAULT_NAME = "DuelQLeapNet"


class DuelQLeapNet(DeepQAgent):
    def __init__(self,
                 action_space,
                 name="DeepQAgent",
                 lr=1e-3,
                 learning_rate_decay_steps=3000,
                 learning_rate_decay_rate=0.99,
                 store_action=True,
                 istraining=False,
                 nb_env=1,
                 **kwargs_converters):
        DeepQAgent.__init__(self, action_space, name, lr, learning_rate_decay_steps, learning_rate_decay_rate,
                            store_action, istraining, nb_env,
                            **kwargs_converters)
        self.tau_dim_start = None
        self.tau_dim_end = None
        self.add_tau = -1  # remove one to tau to have a vector of 0 and 1 instead of 1 and 2
        self._tmp_obs = None

    def init_deep_q(self, transformed_observation):
        self.deep_q = DuelQLeapNet_NN(self.action_space.size(),
                                      observation_size=transformed_observation.shape[-1],
                                      tau_dim_start=self.tau_dim_start,
                                      tau_dim_end=self.tau_dim_end,
                                      add_tau=self.add_tau,
                                      lr=self.lr,
                                      learning_rate_decay_rate=self.learning_rate_decay_rate,
                                      learning_rate_decay_steps=self.learning_rate_decay_steps)

    # grid2op.Agent interface
    def convert_obs(self, observation):
        """
        we added in this file the convert_obs function because it kind of fix the tau_dim_start and tau_dim_end
        too
        """
        if self._tmp_obs is None:
            tmp = np.concatenate(((observation.day_of_week / 7., ),
                                  (observation.hour_of_day / 24., ),
                                  (observation.minute_of_hour / 60., ),
                                           observation.prod_p / observation.gen_pmax,
                                           observation.prod_v / observation.gen_pmax,
                                           observation.load_p / 10.,
                                           observation.load_q / 10.,
                                           observation.actual_dispatch / observation.gen_pmax,
                                           observation.target_dispatch / observation.gen_pmax,
                                           observation.rho,
                                           observation.timestep_overflow,
                                           observation.line_status,
                                           observation.topo_vect,
                                           observation.time_before_cooldown_line / 10.,
                                           observation.time_before_cooldown_sub / 10.,
                                           )).reshape(1, -1)

            # i just want to use the topo_vect as the "tau" in the leap net
            self.tau_dim_start = 3 + observation.prod_p.shape[0] + observation.prod_v.shape[0] + observation.load_p.shape[0]
            self.tau_dim_start = observation.load_q.shape[0] + observation.rho.shape[0]
            self.tau_dim_start += observation.actual_dispatch.shape[0] + observation.target_dispatch.shape[0]
            self.tau_dim_start += observation.timestep_overflow.shape[0] + observation.line_status.shape[0]
            self.tau_dim_end = self.tau_dim_start
            self.tau_dim_end += observation.topo_vect.shape[0]
            self._tmp_obs = np.zeros((1, tmp.shape[1]), dtype=np.float32)
        # TODO optimize that
        self._tmp_obs[:] = np.concatenate(((observation.day_of_week / 7., ),
                                           (observation.hour_of_day / 24., ),
                                           (observation.minute_of_hour / 60., ),
                                           observation.prod_p / observation.gen_pmax,
                                           observation.prod_v / observation.gen_pmax,
                                           observation.load_p / 10.,
                                           observation.load_q / 10.,
                                           observation.actual_dispatch / observation.gen_pmax,
                                           observation.target_dispatch / observation.gen_pmax,
                                           observation.rho,
                                           observation.timestep_overflow,
                                           observation.line_status,
                                           observation.topo_vect,
                                           observation.time_before_cooldown_line / 10.,
                                           observation.time_before_cooldown_sub / 10.,
                                           )).reshape(1, -1)
        return self._tmp_obs
