# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
try:
    from grid2op.Agent import BaseAgent
    from alphaDeesp.expert_operator import expert_operator
    from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation, score_changes_between_two_observations
    from grid2op.Reward import BaseReward, L2RPNReward
    import numpy as np
    import pandas as pd
    import logging
except ImportError as exc_:
    raise ImportError("ExpertAgent baseline impossible to load the required dependencies for using the model. "
                      "The error was: \n {}".format(exc_))


class ExpertAgent(BaseAgent):
    """
    This is an Expert System Agent which tries to solve an overload when it happens and which does not require any
    training.
    For any new overloaded situations, it computes an influence graph around the overload of interest, and rank the
    substations and topologies to explore to find a solution.
    It simulates the top ranked topologies to eventually give a score of success:

    #4 - it solves all overloads,
    #3 - it solves only the overload of interest
    #2 - it partially solves the overload of interest
    #1 - it solves the overload of interest but worsen other overloads
    #0 - it fails

    You can tune:

      - the number of simulations it is allowed to run each for each overload
      - the number of overload you "study" at each given time if there are multiple overloads
      - if you decide to take an action now with a score of 1 (which is not necessarily good or bad, there is a
        tradeoff) or delay it

    """

    def __init__(self,
                 action_space,
                 observation_space,
                 name, gridName="IEEE118",
                 **kwargs):
        super().__init__(action_space)
        self.name = name
        self.grid = gridName  # IEEE14,IEEE118_R2 (WCCI or Neurips Track Robustness), IEEE118
        logging.info("the grid you indicated to the Expert System is:" + gridName)
        self.curr_iter = 0
        self.sub_2nodes = set()
        self.lines_disconnected = set()
        self.action_space = action_space
        self.observation_space = observation_space
        self.threshold_powerFlow_safe = 0.95
        self.maxOverloadsAtATime = 3  # We should not run it more than
        self.config = {
            "totalnumberofsimulatedtopos": 25,
            "numberofsimulatedtopospernode": 5,
            "maxUnusedLines": 2,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2
        }
        self.reward_type = "MinMargin_reward"  # "MinMargin_reward"#we use the L2RPN reward to score the topologies, not the interal alphadeesp score

    def act(self, observation, reward, done=False):
        """
            Parameters
            ----------
            observation: :class:`grid2op.Observation.Observation`
                The current observation of the :class:`grid2op.Environment.Environment`

            reward: ``float``
                The current reward. This is the reward obtained by the previous action

            done: ``bool``
                Whether the episode has ended or not. Used to maintain gym compatibility

            Returns
            -------
            res: :class:`grid2op.Action.PlayableAction`
                The action chosen by the agent.

        """
        self.curr_iter += 1

        # Look for overloads and rank them
        ltc_list = self.getRankedOverloads(observation)
        counterTestedOverloads = 0

        n_overloads = len(ltc_list)
        if n_overloads == 0:  # if no overloads

            if (len(self.sub_2nodes) != 0):  # if any substation is not in the original topology, we will try to get back to it if it is safe
                for sub_id in self.sub_2nodes:
                    action = self.recover_reference_topology(observation, sub_id)
                    if action is not None:
                        return action
            # or we try to reconnect a line if possible
            action = self.reco_line(observation)
            if action is not None:
                return action
            else:
                return self.action_space({})
        # otherwise, we try to solve it
        else:
            best_action = self.action_space({})  # instantiate an action with do nothing action
            subID_ToSplitOn = -1
            scoreBestAction = 0  # range from 0 to 4, 4 is best
            efficacy_best_action = -999
            # ltc_considered_for_action=-1
            subsInCooldown = [i for i in range(observation.n_sub) if (observation.time_before_cooldown_sub[
                                                                          i] >= 1)]  # if you can not run an action currently on a susbtation, so no need to simulate on it

            timestepsOverflowAllowed = observation._obs_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED
            isManyOverloads = (n_overloads > timestepsOverflowAllowed)

            ltcAlreadyConsidered = []

            for ltc in ltc_list:

                isOverflowCritical = (observation.timestep_overflow[ltc] == timestepsOverflowAllowed)
                if (isOverflowCritical) or (ltc not in ltcAlreadyConsidered):
                    ltcAlreadyConsidered.append(ltc)
                    # current_timestep=self.env.chronics_handler.real_data.curr_iter
                    logging.info('running Expert Agent on line with id:' + str(ltc) + ' at timestep:' + str(self.curr_iter))

                    additionalLinesToCut, linesConsidered = self.additionalLinesToCut(
                        ltc)  # for IEEE118_R2, rather than considering one overload, we consider an overloaded corridor of multiple lines
                    ltcAlreadyConsidered += linesConsidered
                    simulator = Grid2opSimulation(observation, self.action_space, self.observation_space, param_options=self.config, debug=False,
                                                  ltc=[ltc], reward_type=self.reward_type)
                    logging.info("doing simulations")
                    ranked_combinations, expert_system_results, actions = expert_operator(simulator, plot=False, debug=False)

                    if (expert_system_results is not None) and (expert_system_results.shape[0] >= 1) and not (
                    expert_system_results["Efficacity"].isnull().values.all()):  # if not empty

                        # we take the best action from the result of the expert system
                        New_scoreBestAction = expert_system_results['Topology simulated score'].max()
                        index_best_action = expert_system_results[
                            expert_system_results['Topology simulated score'] == New_scoreBestAction]["Efficacity"].idxmax()  #

                        logging.info("overloaded line id")
                        logging.info(ltc)
                        logging.info("index_best_action")
                        logging.info(index_best_action)
                        logging.info("New_score_best_action")
                        logging.info(New_scoreBestAction)

                        if ((not np.isnan(index_best_action)) & (New_scoreBestAction > scoreBestAction) & (New_scoreBestAction >= 3)):
                            best_action = actions[index_best_action]
                            efficacy_best_action, scoreBestAction, subID_ToSplitOn = \
                            expert_system_results[['Efficacity', 'Topology simulated score', 'Substation ID']].iloc[index_best_action]

                        if (scoreBestAction == 4):  # we have our good action here that solves all overloads, no need to search further
                            # ltc_considered_for_action=ltc
                            break
                            # TO DO: not necessarily a substation but also line disconnections possibly to consider

                        if ((scoreBestAction == 3) & isOverflowCritical):  # it is a good action to solve our critical overload
                            # ltc_considered_for_action = ltc
                            break

                        # case 1: the overload has no timestep_overflow left, we solve it anyway (score 1) and try to create new overloads
                        # case 2: we have many overloads to solve and it is maybe better to start solving some even with a score of 1
                        if (isOverflowCritical) or ((isManyOverloads) and (scoreBestAction == 0)):
                            indexActionToKeep = self.get_action_with_least_worsened_lines(expert_system_results, ltc_list)
                            ltc_considered_for_action = ltc
                            if (indexActionToKeep is not None):
                                best_action = actions[indexActionToKeep]
                                efficacy_best_action, scoreBestAction, subID_ToSplitOn = \
                                expert_system_results[['Efficacity', 'Topology simulated score', 'Substation ID']].iloc[indexActionToKeep]
                                subID_ToSplitOn = int(subID_ToSplitOn)
                            if (isOverflowCritical):
                                break
                    counterTestedOverloads += 1
                    if (self.maxOverloadsAtATime == counterTestedOverloads):
                        break

            # when the expert system results are not very satisfying, we try some more simulations to merge back nodes in the reference topology for substations with 2 nodes
            # we see if it has a better effect
            if (scoreBestAction <= 1):
                # if (scoreBestAction==0):#get efficacity from do nothing to assess if it is perhaps better to do an action
                #    efficacity = info["rewards"][self.reward_type]
                action = None
                subs_expert_system_results = []
                # in case of IEEE14 grid, disconnecting line 14 is a good action.
                # #The expert system would guess it if we had the ability to look for line disconnections as well

                if (self.grid == "IEEE14"):
                    logging.info("TRYING L 14 Disconnection!!!!!!!")
                    action = self.bonus_action_IEEE14(simulator, scoreBestAction, efficacy_best_action, isOverflowCritical)

                # we will try to get back to initial topology if possible for susbstations considered by the expert system
                if (action is None) and (expert_system_results is not None):
                    subs_expert_system_results = expert_system_results["Substation ID"]
                    action = self.try_out_reference_topologies(simulator, scoreBestAction, efficacy_best_action, isOverflowCritical,
                                                               subs_expert_system_results, subsInCooldown)

                if action is None:  # try all other two nodes substations
                    subs_to_try = set(self.sub_2nodes) - set(subs_expert_system_results)
                    action = self.try_out_reference_topologies(simulator, scoreBestAction, efficacy_best_action, isOverflowCritical,
                                                               subs_to_try, subsInCooldown)
                # if action is None:  # try out overload disconnections
                #   action=self.try_out_overload_disconnections(simulator, scoreBestAction,efficacy_best_action, isOverflowCritical,ltc_list)

                if action is not None:
                    best_action = action
                    subID_ToSplitOn = -1  # this is used to know if a substation will be splitted in 2. In that case we merge 2 nodes, so we set this variable back to default value

            if (subID_ToSplitOn != -1):
                self.sub_2nodes.add(int(subID_ToSplitOn))
            logging.info("action we take is:")
            logging.info(best_action)
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

    # we order overloads by usage rate but also by criticity giving remaining timesteps for overload before disconnect
    def getRankedOverloads(self, observation):
        timestepsOverflowAllowed = observation._obs_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        sort_rho = -np.sort(-observation.rho)  # sort in descending order for positive values
        sort_indices = np.argsort(-observation.rho)
        ltc_list = [sort_indices[i] for i in range(len(sort_rho)) if sort_rho[i] >= 1]

        # now reprioritize ltc if critical or not
        ltc_critical = [l for l in ltc_list if (observation.timestep_overflow[l] == timestepsOverflowAllowed)]
        ltc_not_critical = [l for l in ltc_list if (observation.timestep_overflow[l] != timestepsOverflowAllowed)]

        ltc_list = ltc_critical + ltc_not_critical
        return ltc_list

    # we reconnect lines that were in maintenance or attacked when possible
    def reco_line(self, observation):
        # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if np.any(can_be_reco):
            actions = [self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0]]
            action = actions[0]

            osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)
            if (np.all(osb_simu.rho < self.threshold_powerFlow_safe)) & (len(_info['exception']) == 0):
                return action
        return None

    # for a substation we get the reference topology action
    def reference_topology_sub_action(self, observation, sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        topo_target = list(np.ones(len(topo_vec_sub)).astype('int'))
        action_def = {"set_bus": {"substations_id": [(sub_id, topo_target)]}}
        action = self.action_space(action_def)
        return action

    def recover_reference_topology(self, observation, sub_id):
        topo_vec_sub = observation.state_of(substation_id=sub_id)['topo_vect']
        if (np.any(topo_vec_sub == 2)):
            action = self.reference_topology_sub_action(observation, sub_id)
            # we simulate the action to see if it is safe
            osb_simu, _reward, _done, _info = observation.simulate(action, time_step=0)

            if (np.all(osb_simu.rho < self.threshold_powerFlow_safe)) & (len(_info['exception']) == 0):
                self.sub_2nodes.discard(sub_id)
                return action

        return None

    # we try actions disconnecting overloaded lines
    def try_out_overload_disconnections(self, simulator, scoreBestAction, efficacy_best_action, isOverflowCritical, ltc_list):
        new_ranked_combinations = []

        for l in ltc_list:
            sub_id = simulator.obs.line_or_to_subid[l]
            topo = [l]  # only an array with the line
            new_ranked_combinations.append(pd.DataFrame({
                "score": 1,
                "topology": [topo],
                "node": sub_id
            }))
        isLineDisconnection = True
        action = self.compute_score_on_new_combinations(simulator, new_ranked_combinations, scoreBestAction, efficacy_best_action, isOverflowCritical,
                                                        isLineDisconnection)
        if action is not None:
            logging.info("check")
        return action

    # we try actions merging back a substation in its reference topology
    def try_out_reference_topologies(self, simulator, scoreBestAction, efficacy_best_action, isOverflowCritical, subs_expert_system_results,
                                     subsInCooldown):
        # we will try to get back to initial topology if possible for susbstations considered by the expert system
        new_ranked_combinations = []
        for sub_id in self.sub_2nodes:
            if (sub_id in subs_expert_system_results) and (sub_id not in subsInCooldown):
                # we should pass that to the expert system compute_new_network_changes
                new_ranked_combinations.append(pd.DataFrame({
                    "score": 1,
                    "topology": [simulator.get_reference_topovec_sub(sub_id)],
                    "node": sub_id
                }))
        action = self.compute_score_on_new_combinations(simulator, new_ranked_combinations, scoreBestAction, efficacy_best_action, isOverflowCritical)
        return action

    # for new actions we call the Expert System method that gives a score for a simulation
    def compute_score_on_new_combinations(self, simulator, new_ranked_combinations, scoreBestAction, efficacy_best_action, isOverflowCritical,
                                          isLineDisconnection=False):
        if (len(new_ranked_combinations) >= 1):
            new_expert_system_results, new_actions = simulator.compute_new_network_changes(new_ranked_combinations)

            if (new_expert_system_results.shape[0] >= 1) and not (
                    new_expert_system_results["Efficacity"].isnull().values.all()):  # if not empty
                index_new_best_action = \
                    new_expert_system_results[new_expert_system_results['Topology simulated score'] ==
                                              new_expert_system_results['Topology simulated score'].max()][
                        "Efficacity"].idxmax()

                new_efficacy_best_action, New_scoreBestAction, new_subID_ToActOn = \
                    new_expert_system_results[['Efficacity', 'Topology simulated score', 'Substation ID']].iloc[
                        index_new_best_action]

                if (New_scoreBestAction >= 3) or \
                        ((New_scoreBestAction == 1) and (new_efficacy_best_action >= efficacy_best_action) and (isOverflowCritical)) \
                        or (
                        (New_scoreBestAction >= scoreBestAction) and (new_efficacy_best_action >= efficacy_best_action) and not isLineDisconnection):
                    # 1.01 factor is here for numerical stability

                    best_action = new_actions[index_new_best_action]
                    if isLineDisconnection:
                        line_disconnected = new_expert_system_results["Topology applied"].iloc[index_new_best_action][0]
                        self.lines_disconnected.add(line_disconnected)
                    else:
                        self.sub_2nodes.discard(int(new_subID_ToActOn))
                    return best_action
        return None

    # line 14 is interesting to disconnect in IEEE 14
    def bonus_action_IEEE14(self, simulator, scoreBestAction, efficacy_best_action, isOverflowCritical):
        l = 14
        ltc_list = [l]
        status_l = simulator.obs.line_status[l]
        if (status_l):
            action = self.try_out_overload_disconnections(simulator, scoreBestAction, efficacy_best_action, isOverflowCritical, ltc_list)
            return action
        else:
            return None

    # for critical situations when we need to absolutely solve a given overload
    def get_action_with_least_worsened_lines(self, expert_system_results, ltc_list):
        resultsToConsider = expert_system_results[expert_system_results["Topology simulated score"] == 1]
        nActions = len(resultsToConsider)
        indexActionToKeep = None
        ExistingLinesWorsened = ltc_list
        OtherLinesWorsened = [i for i in range(self.observation_space.n_line)]  # cannot be more lines

        if (nActions != 0):
            for idx in resultsToConsider.index:

                worsened_lines_list = resultsToConsider["Worsened line"][idx]
                RemainingExistingWorsenedLines = set(ltc_list) - set(worsened_lines_list)
                CurrentOtherLinesWorsened = set(worsened_lines_list) - set(ltc_list)
                if len(RemainingExistingWorsenedLines) < len(ExistingLinesWorsened):
                    ExistingLinesWorsened = RemainingExistingWorsenedLines
                    OtherLinesWorsened = CurrentOtherLinesWorsened
                    indexActionToKeep = idx
                elif ((len(RemainingExistingWorsenedLines) == len(ExistingLinesWorsened)) & (
                        len(OtherLinesWorsened) > len(CurrentOtherLinesWorsened))):
                    ExistingLinesWorsened = RemainingExistingWorsenedLines
                    OtherLinesWorsened = CurrentOtherLinesWorsened
                    indexActionToKeep = idx

        return indexActionToKeep

    #to be used when we have lines in parallel that are also overloaded or about to be overloaded
    #The Expert System will compute an overflow graph with those lines cut as well, to tell it that they also belonhg to the constrained path
    def additionalLinesToCut(self,lineToCut):
        additionalLinesToCut=[]
        linesConsidered=[]

        linesToConsider=[]
        if(self.grid=="IEEE118_R2"):
            linesToConsider=[22,23,33,35,34,32]
            pairs=[(22,23),(33,35),(34,32)]

        if (self.grid == "IEEE118"):
            linesToConsider = [135, 136, 149, 147, 148, 146]
            pairs = [(135, 136), (149, 147), (148, 146)]

        if (lineToCut in linesToConsider):
            logging.info("TRYING Multi Line  Disconnection for IEEE118_R2!!!!!!!")
            for p in pairs:
                if (lineToCut in p):
                    additionalLinesToCut = [l for l in p if l != lineToCut]
                    linesConsidered = linesToConsider
                    break

        return additionalLinesToCut, linesConsidered


# using a specific reward to assess the efficacy of an action for the Expert System, in addition to the score 0 to 4
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
        obs = env.current_obs
        if not is_done and not has_error:
            res = np.round(np.min(1.0 - obs.rho), decimals=2)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res


# this dictionnary will help you gather more information on your agent.
other_rewards = {"MinMargin_reward": MinMargin_reward, "l2rpn_atc_reward": L2RPNReward}
