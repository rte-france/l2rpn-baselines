# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from grid2op.Agent import BaseAgent


class DoNothing(BaseAgent):
    """
    Do nothing agent of grid2op, as a lowerbond baseline for l2rpn competition.
    """
    def __init__(self,
                 action_space,
                 observation_space,
                 name,
                 **kwargs):
        super().__init__(action_space)
        self.name = name

    def act(self, observation, reward, done):
        # Just return an empty action aka. "do nothing"
        return self.action_space({})

    def reset(self, observation):
        # No internal states to reset
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass

