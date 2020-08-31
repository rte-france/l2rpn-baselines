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
from grid2op.Reward import BaseReward, L2RPNReward
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
            "totalnumberofsimulatedtopos": 25,#30,
            "numberofsimulatedtopospernode": 5,#10,
            "maxUnusedLines": 3,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }
        self.reward_type="MinMargin_reward"#we use the L2RPN reward to score the topologies, not the interal alphadeesp score

    def act(self, observation, reward, done=False):
        #ltc = None
        #rho_max = 0
        self.curr_iter+=1
        # Look for an overload
        sort_rho=-np.sort(-observation.rho)#sort in descending order for positive values
        sort_indices=np.argsort(-observation.rho)
        ltc_list=[sort_indices[i] for i in range(len(sort_rho)) if sort_rho[i]>=1 ]

        #check substations in cooldown
        if(np.sum(observation.time_before_cooldown_sub)>=1):
            print("cooldown")


        if len(ltc_list)==0:
            topo_vec=observation.topo_vect
            isNotOriginalTopo=np.any(observation.topo_vect == 2)

            if(len(self.sub_2nodes)!=0):# if any substation is not in the original topology, we will try to get back to it if it is safe
                for sub_id in self.sub_2nodes:
                    action=self.recover_reference_topology(observation,sub_id)
                    if action is not None:
                            return action

            return self.action_space({})
        # otherwise, we try to solve it
        else:
            best_action = self.action_space({})  # do nothing action
            subID_ToActOn = -1
            scoreBestAction=0#range from 0 to 4, 4 is best

            for ltc in ltc_list:
                #current_timestep=self.env.chronics_handler.real_data.curr_iter
                print('running Expert Agent on line with id:'+str(ltc)+' at timestep:'+str(self.curr_iter))
                simulator = Grid2opSimulation(observation, self.action_space, self.observation_space, param_options=self.config, debug=False, ltc=[ltc],reward_type=self.reward_type)
                print("doing simulations")
                ranked_combinations, expert_system_results, actions = expert_operator(simulator, plot=False, debug=False)
                # Retreive the line with best score, then best Efficacity
                #print(actions)
                #if(expert_system_results["Efficacity"].isnull().values.all()):
                #    print("check")

                if (expert_system_results.shape[0]>=1) and not (expert_system_results["Efficacity"].isnull().values.all()):#if not empty
                    index_best_action = expert_system_results[
                        expert_system_results['Topology simulated score'] == expert_system_results['Topology simulated score'].max()
                        ]["Efficacity"].idxmax()

                    New_scoreBestAction=expert_system_results['Topology simulated score'][index_best_action]
                    print("overloaded line id")
                    print(ltc)
                    print("index_best_action")
                    print(index_best_action)
                    print("New_score_best_action")
                    print(New_scoreBestAction)


                    if ((not np.isnan(index_best_action)) &(New_scoreBestAction>scoreBestAction)&(New_scoreBestAction>=3)):
                        best_action = actions[index_best_action]
                        subID_ToActOn=int(expert_system_results["Substation ID"][index_best_action])
                        scoreBestAction=New_scoreBestAction
                    if(scoreBestAction==4):#we have our good action here, no need to search further
                        break
                        #TO DO: not necessarily a substation but also line disconnections possibly to consider
                    isOverflowCritical=(observation.timestep_overflow[ltc]==self.observation_space.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED)

                    if ((scoreBestAction == 3) & isOverflowCritical ):
                        break

                    # case 1: the overload has no timestep_overflow left, we solve it anyway (score 1) and try to create new overloads
                    if (isOverflowCritical):
                        indexActionToKeep = self.get_action_with_least_worsened_lines(expert_system_results, ltc_list)
                        if (indexActionToKeep is not None):
                            best_action = actions[indexActionToKeep]
                            subID_ToActOn = int(expert_system_results["Substation ID"][indexActionToKeep])
                            break

                if (scoreBestAction <= 1):
                # we will try to get back to initial topology if possible for susbstations considered by the expert system
                    for sub_id in self.sub_2nodes:
                        if(sub_id in expert_system_results["Substation ID"]):
                            topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
                            if (np.any(topo_vec_sub == 2)):
                                best_action = self.reference_topology_sub_action(observation, sub_id)
                                #subID_ToActOn = sub_id#only when we have a action that leads to 2 nodes
                                break
            if(subID_ToActOn!=-1):
                self.sub_2nodes.add(subID_ToActOn)
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

    def reference_topology_sub_action(self, observation, sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        topo_target = list(np.ones(len(topo_vec_sub)))
        action_def = {"set_bus": {"substations_id": [(sub_id, topo_target)]}}
        action = self.action_space(action_def)
        return action

    def recover_reference_topology(self,observation,sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        if (np.any(topo_vec_sub == 2)):
            action = self.reference_topology_sub_action(observation, sub_id)
            # we simulate the action to see if it is safe
            osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)

            if (np.all(osb_simu.rho < self.threshold_powerFlow_safe)) & (len(_info['exception'])==0):
                self.sub_2nodes.discard(sub_id)
                return action

        return None

    #for critical situations when we need to absolutely solve a given overload
    def get_action_with_least_worsened_lines(self, expert_system_results,ltc_list):
        resultsToConsider = expert_system_results[expert_system_results["Topology simulated score"] == 1]
        nActions = len(resultsToConsider)
        indexActionToKeep=None
        ExistingLinesWorsened=ltc_list
        OtherLinesWorsened=[i for i in range(self.observation_space.n_line)]#cannot be more lines

        if (nActions != 0):
            for idx in resultsToConsider.index:

                worsened_lines_list=resultsToConsider["Worsened line"][idx]
                RemainingExistingWorsenedLines=set(ltc_list)-set(worsened_lines_list)
                CurrentOtherLinesWorsened=set(worsened_lines_list)-set(ltc_list)
                if len(RemainingExistingWorsenedLines)<len(ExistingLinesWorsened):
                    ExistingLinesWorsened=RemainingExistingWorsenedLines
                    OtherLinesWorsened=CurrentOtherLinesWorsened
                    indexActionToKeep=idx
                elif ((len(RemainingExistingWorsenedLines)==len(ExistingLinesWorsened)) &(len(OtherLinesWorsened)>len(CurrentOtherLinesWorsened))):
                    ExistingLinesWorsened=RemainingExistingWorsenedLines
                    OtherLinesWorsened=CurrentOtherLinesWorsened
                    indexActionToKeep=idx

        return indexActionToKeep


class MinMargin_reward(BaseReward):
    """
    if you want to control the reward used by the envrionment when your agent is being assessed, you need
    to provide a class with that specific name that define the reward you want.

    It is important that this file has the exact name "reward" all lowercase, we apologize for the python convention :-/
    """
    def __init__(self):
        # CAREFULL, USING THIS REWARD WILL PROBABLY HAVE LITTLE INTEREST...
        # You can look at the grid2op documentation to have example on definition of rewards
        # https://grid2op.readthedocs.io/en/v0.9.0/reward.html
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = -1.0
        self.reward_max = 1.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        obs=env.current_obs
        if not is_done and not has_error:
            res = np.min(1.0 - obs.rho)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res


# this dictionnary will help you gather more information on your agent.
other_rewards = {"MinMargin_reward": MinMargin_reward, "l2rpn_atc_reward": L2RPNReward}