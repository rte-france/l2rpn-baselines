#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import argparse
import numpy as np
from grid2op.dtypes import dt_int

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *
from l2rpn_baselines.ExpertAgent import ExpertAgent

from l2rpn_baselines.utils.save_log_gif import save_log_gif

DEFAULT_LOGS_DIR = "./logs-eval/expert-agent-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1

def cli():
    parser = argparse.ArgumentParser(description="Eval baseline DDDQN")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOGS_DIR, type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=DEFAULT_NB_PROCESS, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=DEFAULT_MAX_STEPS, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    return parser.parse_args()

def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             grid="IEEE14", #IEEE14,IEEE118_3 (WCCI or Neurips Track Robustness), IEEE118
             seed=None,
             verbose=False,
             save_gif=False):

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Build runner
    agent = ExpertAgent(env.action_space, env.observation_space, "Template",grid)
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent
                    )

    if seed is not None:
        np.random.seed(seed)
        max_int = np.iinfo(dt_int).max
        env_seeds = list(np.random.randint(max_int, size=int(nb_episode)))
        agent_seeds = list(np.random.randint(max_int, size=int(nb_episode)))

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     env_seeds=env_seeds,
                     agent_seeds=agent_seeds,
                     pbar=True)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)

if __name__ == "__main__":
    # Parse command line
    args = cli()
    # Create dataset env
    env = make(args.data_dir,
               reward_class=RedispReward,
               action_class=TopologyChangeAction,
               other_rewards={
                   "bridge": BridgeReward,
                   "overflow": CloseToOverflowReward,
                   "distance": DistanceReward
               })
    # Call evaluation interface
    evaluate(env,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif)
