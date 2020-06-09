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
                 nn_archi,
                 name="DeepQAgent",
                 store_action=True,
                 istraining=False,
                 nb_env=1,
                 **kwargs_converters):
        DeepQAgent.__init__(self,
                            action_space,
                            nn_archi,
                            name=name,
                            store_action=store_action,
                            istraining=istraining,
                            nb_env=nb_env,
                            **kwargs_converters)
        self.tau_dim_start = None
        self.tau_dim_end = None
        self.add_tau = -1  # remove one to tau to have a vector of 0 and 1 instead of 1 and 2
        self._tmp_obs = None
