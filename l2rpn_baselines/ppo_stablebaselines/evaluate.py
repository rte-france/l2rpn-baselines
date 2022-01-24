# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
from grid2op.Runner import Runner

from l2rpn_baselines.utils.save_log_gif import save_log_gif

from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace

from l2rpn_baselines.ppo_stablebaselines.utils import SB3Agent


def evaluate(env,
             load_path=".",
             name="ppo_stable_baselines",
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):
    
    # load the attributes kept
    my_path = os.path.join(load_path, name)
    if not os.path.exists(load_path):
        os.mkdir(load_path)
    if not os.path.exists(my_path):
        os.mkdir(my_path)
        
    with open(os.path.join(my_path, "obs_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        obs_attr_to_keep = json.load(fp=f)
    with open(os.path.join(my_path, "act_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        act_attr_to_keep = json.load(fp=f)

    # create the action and observation space
    gym_observation_space =  BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
    gym_action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_attr_to_keep)
    
    # create a grid2gop agent based on that (this will reload the save weights)
    full_path = os.path.join(load_path, name)
    grid2op_agent = SB3Agent(env.action_space, gym_action_space, gym_observation_space,
                             nn_path=os.path.join(full_path, name))

    # Build runner
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=grid2op_agent)
    
    # Run the agent on the scenarios
    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)

    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=verbose,
                     **kwargs)

    # Print summary
    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)

    if save_gif:
        if verbose:
            print("Saving the gif of the episodes")
        save_log_gif(logs_path, res)
    return grid2op_agent, res


if __name__ == "__main__":
    import grid2op
    from grid2op.Action import CompleteAction
    from grid2op.Reward import L2RPNReward, EpisodeDurationReward, LinesCapacityReward
    from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace
    from lightsim2grid import LightSimBackend
    from grid2op.Chronics import MultifolderWithCache
    import pdb

    nb_episode = 7
    nb_process = 1
    verbose = True

    env = grid2op.make("educ_case14_storage",
                       test=True,
                       action_class=CompleteAction,
                       reward_class=LinesCapacityReward,
                       backend=LightSimBackend())

    evaluate(env,
             nb_episode=nb_episode,
             load_path="./saved_model", 
             name="test4",
             nb_process=1,
             verbose=verbose,
             )

    # to compare with do nothing
    runner_params = env.get_params_for_runner()
    runner = Runner(**runner_params)

    res = runner.run(nb_episode=nb_episode,
                     nb_process=nb_process
                     )

    # Print summary
    if verbose:
        print("Evaluation summary for DN:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)