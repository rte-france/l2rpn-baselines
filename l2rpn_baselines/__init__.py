# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

all_baselines_li = [
    "Template",
    "DoubleDuelingDQN",
    "DoubleDuelingRDQN",
    "DoNothing",
    "ExpertAgent",
    "SliceRDQN",
    "DeepQSimple",
    "DuelQSimple",
    "LeapNetEncoded",
    # Backward compatibility
    "SACOld",
    # contribution
    "PandapowerOPFAgent",
    "Geirina",
    "AsynchronousActorCritic",
    "Kaist",
    # utilitary scripts
    "utils"
]
__version__ = "0.5.1"
