# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from l2rpn_baselines.utils import DeepQAgent
from l2rpn_baselines.SAC.SAC_NN import SAC_NN
DEFAULT_NAME = "SAC"


class SAC(DeepQAgent):
    """
    This is the :class:`l2rpn_baselines.utils` agent representing the SAC agent. This does nothing in particular.
    """
    pass
