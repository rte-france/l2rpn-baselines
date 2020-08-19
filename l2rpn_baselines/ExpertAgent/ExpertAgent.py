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
import numpy as np


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
        self.sub_2nodes=set()
        self.action_space = action_space
        self.observation_space = observation_space
        self.threshold_powerFlow_safe = 0.95
        self.config = {
            "totalnumberofsimulatedtopos": 30,
            "numberofsimulatedtopospernode": 10,
            "maxUnusedLines": 3,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }

    def act(self, observation, reward, done=False):
        ltc = None
        rho_max = 0
        self.curr_iter+=1
        # Look for an overload
        for i, rho in enumerate(observation.rho):
            if rho > 1 and rho > rho_max:
                rho_max = rho
                ltc = i
        # If we find none, do nothing
        if ltc is None:
            topo_vec=observation.topo_vect
            isNotOriginalTopo=np.any(observation.topo_vect == 2)

            if(len(self.sub_2nodes)!=0):# if any substation is not in the original topology, we will try to get back to it if it is safe
                for sub_id in self.sub_2nodes:
                    topo_vec_sub=observation.state_of(substation_id=sub_id)['topo_vect']
                    if(np.any(topo_vec_sub == 2)):
                        topo_target=list(np.ones(len(topo_vec_sub)))
                        action_def = {"set_bus": {"substations_id": [(sub_id, topo_target)]}}
                        action = self.action_space(action_def)
                        #we simulate the action to see if it is safe
                        osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)

                        if np.all(osb_simu.rho < self.threshold_powerFlow_safe):
                            self.sub_2nodes.discard(sub_id)
                            return action

            return self.action_space({})
        # otherwise, we try to solve it
        else:
            #current_timestep=self.env.chronics_handler.real_data.curr_iter
            print('running Expert Agent on line with id:'+str(ltc)+' at timestep:'+str(self.curr_iter))
            simulator = Grid2opSimulation(observation, self.action_space, self.observation_space, param_options=self.config, debug=False, ltc=[ltc])
            print("doing simulations")
            ranked_combinations, expert_system_results, actions = expert_operator(simulator, plot=False, debug=False)
            # Retreive the line with best score, then best Efficacity
            #print(actions)
            
            best_action=self.action_space({})#do nothing action
            if (expert_system_results.shape[0]>=1):#if not empty
                index_best_action = expert_system_results[
                    expert_system_results['Topology simulated score'] == expert_system_results['Topology simulated score'].max()
                    ]["Efficacity"].idxmax()

                print("index_best_action")
                print(index_best_action)


                if not np.isnan(index_best_action):
                    best_action = actions[index_best_action]
                    subID_ToActOn=int(expert_system_results["Substation ID"][index_best_action])
                    self.sub_2nodes.add(subID_ToActOn)
                    #TO DO: not necessarily a substation but also line disconnections possibly to consider

            
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
