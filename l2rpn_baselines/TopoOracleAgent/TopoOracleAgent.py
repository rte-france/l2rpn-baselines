# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

try:
    from oracle4grid.core.agent.OracleAgent import OracleAgent
    _CAN_USE_ORACLE_AGENT = True
except ImportError as exc_:
    _CAN_USE_ORACLE_AGENT = False


class TopoOracleAgent(OracleAgent):
    """
    This is an Oracle Agent that given a user specified reduced action space will compute on a chosen scenario
    all possible action paths up to a certain combinatorial depth between actions.
    It does so with heavy multiprocessing but in a structured and computation efficient manner.
    It is recommended to run its "training" on servers with a lot of CPU cores.
    The computation grows as the size of the action space, the combinatorial depth but also the number of attackable lines:
    attacks are indeed integrated in this computation.

    It finds the best action path aposteriori after running the episode. In that sense, it uses the future to find its path
    and can be regarded as an Oracle. It is not an agent that can be used online as it uses the future and hence proceeds in a deterministic world.
    But it can be used to get upper bounds on what could be achievable in terms of cumulated rewards given the available actions

    You can tune:

      - the "action space" - that is the unitary actions of choice
      - the combinatorial depth
      - the reward considered (and if it should be minimized or maximized)

    You should first "train" your OracleAgent on the chronic of interest which will give you a best action path.
    And then evaluate this agent replaying this action path on the chronic

    """


