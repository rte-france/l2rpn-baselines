# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from grid2op.Agent import BaseAgent
from alphaDeesp.expert_operator import expert_operator
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation


class ExpertAgent(BaseAgent):
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
        self.curr_iter = 0
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = {
            "totalnumberofsimulatedtopos": 10,
            "numberofsimulatedtopospernode": 5,
            "maxUnusedLines": 3,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }

    def act(self, observation, reward, done=False):
        ltc = None
        rho_max = 0
        self.curr_iter += 1
        # Look for an overload
        for i, rho in enumerate(observation.rho):
            if rho > 1 and rho > rho_max:
                rho_max = rho
                ltc = i
        # If we find none, do nothing
        if ltc is None:
            return self.action_space({})
        # otherwise, we try to solve it
        else:
            # current_timestep=self.env.chronics_handler.real_data.curr_iter
            print('running Expert Agent on line with id:' + str(ltc) + ' at timestep:' + str(self.curr_iter))
            simulator = Grid2opSimulation(observation, self.action_space, self.observation_space, param_options=self.config, debug=False, ltc=[ltc])
            ranked_combinations, expert_system_results, actions = expert_operator(simulator, plot=False, debug=False)
            # Retreive the line with best score, then best Efficacity
            index_best_action = expert_system_results[
                expert_system_results['Topology simulated score'] == expert_system_results['Topology simulated score'].max()
                ]["Efficacity"].idxmax()
            best_action = actions[index_best_action]
            print("action we take is:")
            print(best_action)
            return best_action

    def reset(self, observation):
        # No internal states to reset
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass
