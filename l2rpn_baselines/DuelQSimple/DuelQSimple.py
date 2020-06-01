# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from l2rpn_baselines.utils import DeepQAgent
from l2rpn_baselines.DuelQSimple.DuelQ_NN import DuelQ_NN
DEFAULT_NAME = "DuelQSimple"


class DuelQSimple(DeepQAgent):
    def init_deep_q(self, transformed_observation):
        self.deep_q = DuelQ_NN(self.action_space.size(),
                               observation_size=transformed_observation.shape[-1],
                               lr=self.lr,
                               learning_rate_decay_rate=self.learning_rate_decay_rate,
                               learning_rate_decay_steps=self.learning_rate_decay_steps)
